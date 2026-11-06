'''
Model loading and batched prediction for the UI.

Every v20+ model takes the 929 columns of starting_decks.csv (cards with a _u column for upgraded copies, relics, floor, then the four one hot characters). The
v17-v19 models take the same columns without the nine Searing Blow level columns beyond searingblow_u (920). v14-v16 take an older 863 column layout (cards,
relics and the four characters, no floor) that build_legacy_layouts.py rebuilds from git history into data/legacy_layouts.json. Those models were trained on final
decks to predict whether the run won at all (any victory), so the floor does not change their output and it is not a Heart kill probability.

The forward pass is rebuilt from each model's weights, as scripts/interaction_bench.py does, rather than calling the saved TorchScript: the cumulative
upgrade models (v22 *_cum) fold each upgraded copy onto its plain column inside forward, and that saved graph does not run reliably across devices.
v24 and later models also take run state after the 929 columns (see scripts/run_state.py): HP as a percentage of max HP, max HP and gold in units of 25, as the integers
the training files hold (the network rescales them with its input_scale buffer). Which of them a model takes is in experiments.csv (extra_features). A state that gives no
HP or gold gets the neutral values the training data uses for missing ones. v24+ models need real HP and gold to be at their best.
Models output logits for every head, sigmoid gives the probability. v21+ models have five heads (act 1 boss, act 2 boss, act 3 boss, act 3 second boss at A20, Heart kill),
earlier ones just the Heart head. Single target (_st) models only trained the Heart head, so only that output is meaningful.
'''

import glob
import json
import os
import re
import sys
import threading

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, f"{os.getcwd()}/scripts")
from run_state import EXTRA_GROUPS, NEUTRAL # the run state columns and the values used when a run has none

ROOT = os.getcwd()
HERE = os.path.dirname(os.path.abspath(__file__))
PROD = f'{ROOT}/model_objects/production'
HEADS = ['act1_boss', 'act2_boss', 'act3_boss', 'act3_second_boss', 'heart']
SEARING_EXTRA = [f'searingblow_u{n}plus' for n in range(2, 11)]
CHARACTERS = ['ironclad', 'the_silent', 'defect', 'watcher']

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

COLS_929 = list(pd.read_csv(f'{ROOT}/starting_decks.csv', nrows = 0).columns)
LAYOUTS = {929: COLS_929, 920: [c for c in COLS_929 if c not in SEARING_EXTRA]}
LEGACY_PATH = f'{HERE}/data/legacy_layouts.json'
LEGACY = json.load(open(LEGACY_PATH)) if os.path.exists(LEGACY_PATH) else {'candidates': {}, 'models': {}}
LEGACY_LABEL = {'master_lists@0f5490b': 'underscore card names (master lists of 2024-08-27)', 'master_lists@3f8c2e6': 'one word card names (master lists of 2024-09-08)'}

def layout_columns(name, n_in, extras = ()):
    '''Current column name (None where the current data has no such column) at each input position of a model.'''
    if extras:
        return COLS_929 + list(extras)
    if n_in in LAYOUTS:
        return LAYOUTS[n_in]
    info = LEGACY['models'].get(name)
    return LEGACY['candidates'][info['layout']]['columns'] if info and n_in == 863 else None

def _experiments():
    path = f'{ROOT}/model_objects/experiments.csv'
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path).drop_duplicates('model_version', keep = 'last').set_index('model_version')
    return df.to_dict('index')

class Net:
    def __init__(self, path):
        sd = torch.jit.load(path, map_location = 'cpu').state_dict()
        self.base_idx = sd['base_idx'].to(DEVICE) if 'base_idx' in sd else None
        self.up_idx = sd['up_idx'].to(DEVICE) if 'up_idx' in sd else None
        self.scale = sd['input_scale'].to(DEVICE) if 'input_scale' in sd else None
        sd = {k: v for k, v in sd.items() if k.startswith('linear_relu_stack')}
        keys = sorted({k.rsplit('.', 1)[0] for k in sd}, key = lambda s: int(s.split('.')[-1]))
        self.layers = [(sd[k + '.weight'].to(DEVICE), sd[k + '.bias'].to(DEVICE)) for k in keys]
        self.n_in = self.layers[0][0].shape[1]
        self.n_out = self.layers[-1][0].shape[0]

    @torch.no_grad()
    def logits(self, X, batch = 20000):
        out = []
        for i in range(0, len(X), batch):
            x = torch.as_tensor(X[i:i + batch], dtype = torch.float32).to(DEVICE)
            if self.base_idx is not None:
                x = x.index_add(1, self.base_idx, x.index_select(1, self.up_idx))
            if self.scale is not None:
                x = x * self.scale
            for j, (W, b) in enumerate(self.layers):
                x = x @ W.T + b
                if j < len(self.layers) - 1:
                    x = torch.relu(x)
            out.append(x.float().cpu())
        return torch.cat(out).numpy().astype(np.float64)

def model_version(name):
    m = re.match(r'fnn_v(\d+)', name)
    return int(m.group(1)) if m else 0

def _json_list(value):
    return json.loads(value) if isinstance(value, str) else None

def extra_columns(row):
    '''Run state columns a model takes after the 929 columns, in the order train_model.py builds them (groups in the order of extra_features).'''
    groups = _json_list(row.get('extra_features')) or []
    return [c for g in groups for c in EXTRA_GROUPS[g]]

def describe(name, net_shape, experiments):
    '''Registry entry for one model file, from its weights (input / output width) and the experiments log (how it was trained).'''
    n_in, n_out = net_shape
    row = experiments.get(name, {})
    stages = _json_list(row.get('stages'))
    hw = _json_list(row.get('head_weights'))
    extras = extra_columns(row)
    legacy = n_in == 863
    if n_out == 5 and hw is not None:
        heads = [h for h, w in zip(HEADS, hw) if w > 0] # a head with weight 0 was never trained, its output means nothing
    elif n_out == 5:
        heads = HEADS
    else:
        heads = ['heart']
    supported = (n_in == 929 + len(extras) and (extras or n_in in LAYOUTS)) or n_in in LAYOUTS and not extras or (legacy and name in LEGACY['models'])
    enc = row.get('upgrade_encoding') if isinstance(row.get('upgrade_encoding'), str) else ('split' if n_in == 929 else None)
    flip_w = row.get('upgrade_flip_weight')
    flip = None
    if enc == 'cumulative' and isinstance(flip_w, (int, float)) and not pd.isna(flip_w) and flip_w > 0:
        mode = row.get('upgrade_flip_mode')
        flip = mode if isinstance(mode, str) else 'rows'
    return {
        'id': name, 'version': model_version(name), 'inputs': n_in, 'outputs': n_out,
        'supported': bool(supported),
        'reason': None if supported else f'Takes {n_in} input columns, a layout that could not be rebuilt.',
        'floor_input': not legacy, 'target': 'win' if legacy else 'heart',
        'layout_note': (f"Input layout rebuilt from git history: {LEGACY_LABEL.get(LEGACY['models'][name]['layout'], LEGACY['models'][name]['layout'])}, chosen by held-out AUC (gap {LEGACY['models'][name]['auc_gap_vs_other']:.3f} over the other candidate)") if legacy and name in LEGACY['models'] else None,
        'heads': heads,
        'extras': extras, 'uses_hp': 'hp_pct' in extras, 'uses_gold': 'gold_25' in extras,
        'stages': [x.replace('pre_training_', '') for x in stages] if stages else None,
        'upgrade_encoding': enc, 'upgrade_flip': flip,
        'train_minutes': None if row.get('train_minutes') is None or pd.isna(row.get('train_minutes')) else round(float(row['train_minutes'])),
        'git_commit': row.get('git_commit') if isinstance(row.get('git_commit'), str) else None,
    }

def run_state_values(state):
    '''The integer run state columns for one state, by the rules of run_state.py: HP as a percentage of max HP, max HP (1 to 127), gold in units of 25 (up to 3000).'''
    out = dict(NEUTRAL)
    max_hp, hp, gold = state.get('max_hp'), state.get('hp'), state.get('gold')
    if isinstance(max_hp, (int, float)) and max_hp > 0:
        out['max_hp'] = int(min(max(max_hp, 1), 127))
        if isinstance(hp, (int, float)):
            out['hp_pct'] = int(round(100 * min(max(hp, 0), max_hp) / max_hp))
    if isinstance(gold, (int, float)):
        out['gold_25'] = int(min(max(gold, 0), 3000) // 25)
    return out

class Engine:
    def __init__(self):
        self.lock = threading.Lock()
        self.nets = {}
        self.index = {}
        self.experiments = _experiments()
        self.registry = {}
        # Reading a model's layer sizes means opening the file, so they are cached by file size and time (the cache is rebuilt for any file that changed)
        cache_path = f'{HERE}/data/shapes_cache.json'
        cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
        fresh = {}
        for path in sorted(glob.glob(f'{PROD}/*.pth')):
            name = os.path.basename(path)[:-4]
            st = os.stat(path)
            key = f'{st.st_size}_{int(st.st_mtime)}'
            try:
                if name in cache and cache[name]['key'] == key:
                    shape = tuple(cache[name]['shape'])
                else:
                    sd = torch.jit.load(path, map_location = 'cpu').state_dict()
                    ws = [v.shape for k, v in sd.items() if v.ndim == 2 and k.startswith('linear_relu_stack')]
                    shape = (int(ws[0][1]), int(ws[-1][0]))
                fresh[name] = {'key': key, 'shape': list(shape)}
                self.registry[name] = describe(name, shape, self.experiments)
            except Exception as e:
                self.registry[name] = {'id': name, 'version': model_version(name), 'supported': False, 'reason': f'Could not read the file: {e}'.splitlines()[0][:120], 'heads': ['heart']}
        if fresh != cache:
            json.dump(fresh, open(cache_path, 'w'))

    def net(self, name):
        if name not in self.nets:
            info = self.registry.get(name)
            if info is None or not info['supported']:
                raise ValueError(f'model {name} is not available')
            if len(self.nets) >= 3: # keep a few loaded, they are 14 MB each
                self.nets.pop(next(iter(self.nets)))
            self.nets[name] = Net(f'{PROD}/{name}.pth')
        return self.nets[name]

    def predict(self, name, character, states):
        '''
        states: [{floor: int, cards: {column: count}, relics: [column]}]. Returns an array (len(states), n_heads) of probabilities, columns in HEADS order for 5 head models
        (a single Heart column for older ones).
        '''
        if character not in CHARACTERS:
            raise ValueError(f'unknown character {character}')
        with self.lock:
            net = self.net(name)
            if name not in self.index:
                cols = layout_columns(name, net.n_in, self.registry[name].get('extras', ()))
                idx = {}
                for i, c in enumerate(cols):
                    if c is not None:
                        idx.setdefault(c, []).append(i)
                self.index[name] = (len(cols), idx)
            width, idx = self.index[name]
            X = np.zeros((len(states), width), dtype = np.float32)
            for i in idx[character]:
                X[:, i] = 1
            for r, s in enumerate(states):
                for i in idx.get('floor', []):
                    X[r, i] = s['floor']
                if idx.keys() & set(NEUTRAL):
                    for col, v in run_state_values(s).items():
                        for i in idx.get(col, []):
                            X[r, i] = v
                for col, n in s['cards'].items():
                    if col not in COLS_929:
                        raise ValueError(f'unknown card column {col}')
                    for i in idx.get(col, []): # a column the model does not have (Searing Blow levels above +1 in v17-v19, cards the old lists lacked) is left out
                        X[r, i] += n
                for col in s['relics']:
                    if col not in COLS_929:
                        raise ValueError(f'unknown relic column {col}')
                    for i in idx.get(col, []):
                        X[r, i] += 1
            z = net.logits(X)
        return 1 / (1 + np.exp(-z))
