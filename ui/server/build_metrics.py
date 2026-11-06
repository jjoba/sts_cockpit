'''
Writes data/metrics.json (held-out Heart head metrics for every runnable v21+ model) and data/start_run_state.json (the usual HP and gold at the start of a run).
Run from the project root:  python ui/server/build_metrics.py

All models are scored with the UI's own engine on one test set, the finetuning test file of data/training_data_v24 (v24 and later models need its run state columns,
and modded runs are out of it), so every number in the model directory is comparable, v21 and v22 included. Models trained on other splits (v20 and earlier) get none.
Precision, recall and F1 are at the standard 0.5 threshold, ECE uses equal frequency bins (as train_model.py does), log loss is clipped at 1e-7.
As a check, the numbers match the ones in outstanding_experiments.md and model_objects/bench/v24/baselines.csv.
'''

import json
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine
sys.path.insert(0, f'{os.getcwd()}/scripts')
import run_state # START_GOLD: a run starts with 99 gold, the data holds gold in units of 25 (3)

ROOT = os.getcwd()
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = f'{ROOT}/data/training_data_v24'
MIN_VERSION = 21
CHUNK = 200000

def metrics(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    pos = p >= 0.5
    tp = int((pos & (y == 1)).sum())
    precision, recall = tp / max(int(pos.sum()), 1), tp / max(int(y.sum()), 1)
    bins = np.array_split(np.argsort(p), 10)
    return {
        'log_loss': float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 'auc': float(roc_auc_score(y, p)),
        'ece': float(sum(len(c) * abs(p[c].mean() - y[c].mean()) for c in bins) / len(y)), 'brier': float(np.mean((p - y) ** 2)),
        'precision': precision, 'recall': recall, 'f1': 2 * precision * recall / max(precision + recall, 1e-12), 'avg_precision': float(average_precision_score(y, p)),
    }

def main():
    layout = json.load(open(f'{DATA}/layout.json'))
    extra_cols = layout['extra_cols']
    n_cols = layout['n_base'] + len(extra_cols) + layout['n_labels']
    data = np.memmap(f'{DATA}/finetuning/test/test.bin', dtype = np.int8, mode = 'r').reshape(-1, n_cols)
    y = np.asarray(data[:, -1]).astype(np.int8)
    print(f'{len(y)} test rows, Heart rate {y.mean():.4f}', flush = True)

    # Usual start of a run: the most common max HP, HP percentage and gold on floor 1, per character
    floor_col = engine.COLS_929.index('floor')
    start = {}
    for ch in engine.CHARACTERS:
        col = engine.COLS_929.index(ch)
        rows = np.concatenate([np.asarray(data[a:a + CHUNK][:, [floor_col, col] + [layout['n_base'] + i for i in range(len(extra_cols))]]) for a in range(0, len(data), CHUNK)])
        rows = rows[(rows[:, 0] == 1) & (rows[:, 1] == 1)]
        mode = lambda v: int(np.bincount(v.astype(int) - v.astype(int).min()).argmax() + v.astype(int).min())
        start[ch] = {'max_hp': mode(rows[:, 2 + extra_cols.index('max_hp')]), 'hp_pct': mode(rows[:, 2 + extra_cols.index('hp_pct')]), 'gold': run_state.START_GOLD, 'rows': int(len(rows))}
    print('start of run', start, flush = True)
    json.dump(start, open(f'{HERE}/data/start_run_state.json', 'w'), indent = 1)

    E = engine.Engine()
    out = {}
    for name, info in sorted(E.registry.items()):
        if not info['supported'] or info['version'] < MIN_VERSION:
            continue
        idx = list(range(layout['n_base'])) + [layout['n_base'] + extra_cols.index(c) for c in info['extras']]
        net = E.net(name)
        z = np.concatenate([net.logits(np.asarray(data[a:a + CHUNK][:, idx]).astype(np.float32))[:, -1] for a in range(0, len(data), CHUNK)])
        out[name] = metrics(y, 1 / (1 + np.exp(-z)))
        print(name, {k: round(v, 5) for k, v in out[name].items() if k in ('log_loss', 'auc')}, flush = True)
    json.dump({'test_set': {'folder': 'data/training_data_v24/finetuning/test', 'rows': int(len(y)), 'heart_rate': float(y.mean()), 'note': 'finetuning test file with modded runs removed, the same for every model'}, 'models': out},
              open(f'{HERE}/data/metrics.json', 'w'), indent = 1)
    print('metrics for', len(out), 'models')

main()
