'''
Item 14 follow-up: how much do the v24 models depend on HP and gold being supplied? Scores each model on the cleaned finetuning test set with its extra
inputs (a) as logged, (b) replaced by the neutral defaults of run_state.NEUTRAL, (c) replaced by the median of the training rows at the same floor
(what a tool that only knows the floor could supply), (d) set to zero (inputs missing). Models without extras are scored once as the reference.
Run from the project root: python scripts/eval_extras.py <data_dir_name> <out_csv> <model name> [<model name> ...]
'''
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_model as tm
from run_state import EXTRA_COLS, NEUTRAL

root = os.getcwd()
data_name, out_csv, names = sys.argv[1], sys.argv[2], sys.argv[3:]
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
METRICS = ['log_loss', 'auc', 'ece', 'brier', 'precision_at_0.5', 'recall_at_0.5', 'f1_at_0.5', 'avg_precision']

class Replaced:
    # Test set view whose extra columns are overwritten by a fixed row vector (n_extra) or a per floor table (128 x n_extra)
    def __init__(self, base, cols, mode, values):
        self.base, self.cols, self.mode, self.values = base, cols, mode, values
        self.data = base.data

    def __len__(self):
        return len(self.base)

    def take(self, idx):
        rows = torch.from_numpy(np.asarray(self.data[idx])).clone()
        if self.mode == 'per_floor':
            floor = rows[:, tm.FLOOR_COL].long().clamp(0, 127)
            rows[:, self.cols] = torch.from_numpy(self.values)[floor]
        elif self.mode == 'fixed':
            rows[:, self.cols] = torch.from_numpy(self.values)
        return rows[:, tm.INPUT_COLS], rows[:, -tm.N_HEADS:]

rows = []
for name in names:
    cfg_path = f'{root}/model_objects/dev/{name}/config.json'
    extras = json.load(open(cfg_path))['config'].get('extra_features', []) if os.path.exists(cfg_path) else []
    tm.set_layout(f'{root}/data/{data_name}', extras)
    test = tm.BinDataset(f'{root}/data/{data_name}/finetuning/test/test.bin')
    train = tm.BinDataset(f'{root}/data/{data_name}/finetuning/train/train.bin')
    y = np.asarray(test.data[:, -1]).astype(np.int8)
    base_rate = float(np.asarray(train.data[::50, -1]).mean())
    model = torch.jit.load(f'{root}/model_objects/production/{name}.pth', map_location = device)

    layout_cols = json.load(open(f'{root}/data/{data_name}/layout.json'))['extra_cols']
    used = [c for c in EXTRA_COLS if c in [x for g in extras for x in tm.EXTRA_GROUPS[g]]]
    col_idx = [tm.N_FEATURES + layout_cols.index(c) for c in used]

    modes = {'as logged': None}
    if used:
        sample = np.asarray(train.data[::20])
        fl = sample[:, tm.FLOOR_COL].astype(int)
        table = np.zeros((128, len(used)), dtype = np.int8)
        for f in range(128):
            m = fl == f
            table[f] = np.median(sample[m][:, col_idx], axis = 0) if m.sum() >= 50 else np.median(sample[:, col_idx], axis = 0)
        modes['neutral defaults'] = Replaced(test, col_idx, 'fixed', np.array([NEUTRAL[c] for c in used], dtype = np.int8))
        modes['median at floor'] = Replaced(test, col_idx, 'per_floor', table)
        modes['zeros'] = Replaced(test, col_idx, 'fixed', np.zeros(len(used), dtype = np.int8))

    for mode, ds in modes.items():
        ds = test if ds is None else ds
        p = tm.predict(model, ds, np.arange(len(test)), device)[:, tm.HEART]
        m, _ = tm.compute_metrics(y, p, base_rate)
        rows.append({'model': name, 'extras': '+'.join(extras) or 'none', 'extra_values': mode, **{k: m[k] for k in METRICS}})
        print(rows[-1], flush = True)

os.makedirs(os.path.dirname(out_csv), exist_ok = True)
pd.DataFrame(rows).to_csv(out_csv, index = False)
