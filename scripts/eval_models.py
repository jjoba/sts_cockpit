'''
Scores saved models on the finetuning test set of a data folder with the standard metrics (log loss, AUC, ECE, Brier, precision / recall / F1 at 0.5, plus average precision),
so models trained on different data versions can be compared on one test set, e.g. v22 / v23 on the cleaned v24 test set.
A model's inputs are read from model_objects/dev/<name>/config.json (extra_features), models without one take the 929 base features.
Run from the project root: python scripts/eval_models.py <data_dir_name> <out_csv> <model name> [<model name> ...]
'''
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_model as tm

root = os.getcwd()
data_name, out_csv, names = sys.argv[1], sys.argv[2], sys.argv[3:]
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

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
    p = tm.predict(model, test, np.arange(len(test)), device)[:, tm.HEART]
    m, _ = tm.compute_metrics(y, p, base_rate)
    rows.append({'model': name, 'extra_features': '+'.join(extras) or 'none', 'rows': len(y), **{k: m[k] for k in ['log_loss', 'auc', 'ece', 'brier', 'precision_at_0.5', 'recall_at_0.5', 'f1_at_0.5', 'avg_precision']}})
    print(rows[-1], flush = True)

os.makedirs(os.path.dirname(out_csv), exist_ok = True)
new = pd.DataFrame(rows)
if os.path.exists(out_csv):
    new = pd.concat([pd.read_csv(out_csv), new]).drop_duplicates('model', keep = 'last')
new.to_csv(out_csv, index = False)
