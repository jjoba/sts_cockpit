'''
Rebuilds the input layout of the pre-v17 models (fnn_v14 .. fnn_v16, 863 inputs) so the UI can run them. Run from the project root:

    python ui/server/build_legacy_layouts.py

Those models were trained before per floor rows existed: the input is the final deck, with 679 card columns (plain and _u), 180 relic columns and the four one-hot
characters, no floor, and the label was "the run won" (any victory), not Heart kill. The card and relic lists were regenerated since (see git history of
data/master_lists), and on 2024-09-08 the names became one word names, which also changed the order of 68 columns. Which list a given model was trained on is
not recorded, so each candidate list from git is tried: every model is scored under each on held-out rows from floor 40 on (near final decks, which is what
these models were trained on; on early floor rows the two orderings are hard to tell apart) and the layout with the best AUC is kept, with a bootstrap over
runs for the size of the gap. Old column names that were later renamed are mapped to the current column (ALIAS below); columns the current data no longer has stay empty.

Result when this was written: v14, v15, v15_small and v15_large fit the underscore list of 2024-08-27 (AUC gap about 0.03), v16 fits the one word list of 2024-09-08
(gap 0.02), which matches the dates of the model files (v16 was saved the day after the rename).

Writes data/legacy_layouts.json: the columns of each candidate layout (as current column names) and, per model, the layout chosen with the evidence.
'''

import json
import os
import re
import subprocess

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = os.getcwd()
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ['fnn_v14', 'fnn_v15', 'fnn_v15_small', 'fnn_v15_large', 'fnn_v16_w_2_24m_clf']
CHARACTERS = ['defect', 'ironclad', 'the_silent', 'watcher']
N_ROWS = 250000
MIN_FLOOR = 40

# Old column (without _u) -> current column, for cards renamed in the game or merged by the later one word naming
ALIAS = {
    'conserve_battery': 'chargebattery', 'conservebattery': 'chargebattery', 'lockon': 'bullseye', 'steam_power': 'steambarrier', 'steampower': 'steambarrier',
    'underhanded_strike': 'sneakystrike', 'underhandedstrike': 'sneakystrike', 'undo': 'equilibrium', 'wireheading': 'foresight',
    'fasting2': 'fasting', 'wraith_form_v2': 'wraithformv', 'frozen_egg_2': 'frozenegg', 'molten_egg_2': 'moltenegg', 'toxic_egg_2': 'toxicegg',
}

def blob(commit, path):
    return subprocess.run(['git', 'cat-file', 'blob', f'{commit}:{path}'], capture_output = True, text = True, cwd = ROOT).stdout

def candidates():
    # master lists as shipped with v15 (underscore names) and after the one word renaming; the columns are cards, relics, then the characters
    cands = {}
    for c in ['0f5490b', '3f8c2e6']:
        cands[f'master_lists@{c}'] = blob(c, 'data/master_lists/master_card_list.txt').split() + blob(c, 'data/master_lists/master_relic_list.txt').split() + CHARACTERS
    return {k: v for k, v in cands.items() if len(v) == 863}

def norm(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())

def main():
    cur = list(pd.read_csv(f'{ROOT}/starting_decks.csv', nrows = 0).columns)
    cur_key = {}
    for i, c in enumerate(cur):
        u = c.endswith('_u')
        cur_key.setdefault(norm(c[:-2] if u else c) + ('|u' if u else ''), i)

    def to_current(name):
        u = name.endswith('_u')
        base = name[:-2] if u else name
        base = ALIAS.get(base, base)
        return cur_key.get(norm(base) + ('|u' if u else ''))

    cands = candidates()
    maps = {k: [to_current(n) for n in v] for k, v in cands.items()}
    n_cols = len(cur) + 5
    data = np.memmap(f'{ROOT}/data/training_data/finetuning/test/test.bin', dtype = np.int8, mode = 'r').reshape(-1, n_cols)
    floor_col = cur.index('floor')
    late = []
    for start in range(0, len(data), 2000000):
        block = np.asarray(data[start:start + 2000000])
        late.append(block[block[:, floor_col] >= MIN_FLOOR])
    rows = np.concatenate(late)
    rng = np.random.default_rng(0)
    rows = rows[np.sort(rng.choice(len(rows), min(len(rows), N_ROWS), replace = False))]
    y = rows[:, -1]
    Xc = np.concatenate([rows[:, :len(cur)].astype(np.float32), np.zeros((len(rows), 1), np.float32)], axis = 1) # last column is all zero, for columns with no match
    run = np.cumsum(np.r_[0, np.diff(rows[:, floor_col].astype(int)) < 0]) # rows of one run are adjacent with rising floors
    run_rows = [np.where(run == r)[0] for r in np.unique(run)]
    print(f'{len(rows)} rows from floor {MIN_FLOOR} on, Heart kill rate {y.mean():.3f}, {len(run_rows)} runs', flush = True)
    X = {k: Xc[:, [len(cur) if i is None else i for i in m]] for k, m in maps.items()}

    out = {'candidates': {k: {'columns': [None if i is None else cur[i] for i in m], 'unmatched': [n for n, i in zip(cands[k], m) if i is None]} for k, m in maps.items()}, 'models': {}}
    names = list(X)
    for name in MODELS:
        net = torch.jit.load(f'{ROOT}/model_objects/production/{name}.pth', map_location = 'cpu').eval()
        pred = {}
        for k, x in X.items():
            with torch.no_grad():
                pred[k] = net(torch.as_tensor(x))[:, -1].numpy()
        aucs = {k: float(roc_auc_score(y, p)) for k, p in pred.items()}
        best = max(aucs, key = aucs.get)
        other = [k for k in names if k != best][0]
        gaps = []
        for _ in range(40):
            ix = np.concatenate([run_rows[r] for r in rng.choice(len(run_rows), min(3000, len(run_rows)))])
            gaps.append(roc_auc_score(y[ix], pred[best][ix]) - roc_auc_score(y[ix], pred[other][ix]))
        out['models'][name] = {'layout': best, 'auc_by_layout': aucs, 'auc_gap_vs_other': float(np.mean(gaps)), 'gap_5_95': [float(v) for v in np.percentile(gaps, [5, 95])]}
        print(name, {k: round(v, 4) for k, v in aucs.items()}, '->', best, 'gap', round(float(np.mean(gaps)), 4), flush = True)
    out['note'] = f'AUC against the Heart kill label of held-out finetuning rows from floor {MIN_FLOOR} on. The models were trained on "won the run" and on final decks, so these numbers only serve to tell the layouts apart.'
    json.dump(out, open(f'{HERE}/data/legacy_layouts.json', 'w'), indent = 1)

main()
