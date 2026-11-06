'''
Rewrites the "v24 results" table in outstanding_experiments.md (between the v24-results markers) from the baseline scores
(model_objects/bench/v24/baselines.csv, v22 / v23 scored on the cleaned v24 test set) and the fnn_v24 rows of model_objects/experiments.csv.
Run from the project root after each model: python scripts/v24_record.py
'''
import os
import pandas as pd

START, END = '<!-- v24-results-start -->', '<!-- v24-results-end -->'
root = os.getcwd()
cols = [('log_loss', 'Log loss', '{:.5f}'), ('auc', 'AUC', '{:.4f}'), ('ece', 'ECE', '{:.5f}'), ('brier', 'Brier', '{:.6f}'),
        ('precision_at_0.5', 'Precision@0.5', '{:.3f}'), ('recall_at_0.5', 'Recall@0.5', '{:.3f}'), ('f1_at_0.5', 'F1@0.5', '{:.3f}'), ('avg_precision', 'Avg precision', '{:.4f}')]

rows = []
base_path = f'{root}/model_objects/bench/v24/baselines.csv'
if os.path.exists(base_path):
    for _, r in pd.read_csv(base_path).iterrows():
        rows.append([f"{r['model']} (re-scored on v24 test set)", 'none', '-'] + [f.format(r[c]) for c, _, f in cols] + ['-'])

exp_path = f'{root}/model_objects/experiments.csv'
if os.path.exists(exp_path):
    e = pd.read_csv(exp_path)
    for _, r in e[e.model_version.astype(str).str.match(r'fnn_v2[4-9]')].iterrows():
        extra = '+'.join(eval(r['extra_features'].replace('true', 'True')) if isinstance(r.get('extra_features'), str) and r['extra_features'].startswith('[') else []) or 'none'
        vals = [(f.format(r[f'best_ft_{c}']) if f'best_ft_{c}' in r and pd.notna(r[f'best_ft_{c}']) else 'n/a') for c, _, f in cols]
        w = r.get('upgrade_flip_weight')
        flip = 'none' if pd.isna(w) or w == 0 else f"{r['upgrade_flip_mode'] if isinstance(r.get('upgrade_flip_mode'), str) else 'rows'} w{w:g} every {int(r['upgrade_flip_every']) if pd.notna(r.get('upgrade_flip_every')) else 1} rows {int(r['upgrade_flip_rows'])}" + (f"+{int(r['upgrade_flip_synth_rows'])}" if r.get('upgrade_flip_mode') == 'synthetic' and pd.notna(r.get('upgrade_flip_synth_rows')) else '')
        rows.append([r['model_version'], extra, flip] + vals + [f"{r['train_minutes']:.0f} min"])

header = ['Model', 'Extra inputs', 'Flip'] + [h for _, h, _ in cols] + ['Train time']
table = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)] + ['| ' + ' | '.join(r) + ' |' for r in rows]

p = f'{root}/outstanding_experiments.md'
s = open(p).read()
block = f'{START}\n' + '\n'.join(table) + f'\n{END}'
if START in s:
    s = s[:s.index(START)] + block + s[s.index(END) + len(END):]
else:
    raise SystemExit('v24-results markers missing in outstanding_experiments.md')
open(p, 'w').write(s)
print('\n'.join(table))
