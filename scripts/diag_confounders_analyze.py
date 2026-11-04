'''
Item 12, stage 1 (analyze): which candidate confounders differ between plain-card and upgraded-card rows, and how much does stratifying on each
move the raw upgraded-vs-plain association (Mantel-Haenszel log odds ratio, same as diag_upgrades.py, train rows only). Also how predictive each candidate is
on its own (logistic baselines, train fit, test evaluation). No neural network involved.
Input: output of diag_confounders_extract.py. Run from the project root: python scripts/diag_confounders_analyze.py <extract_dir> <out_dir>
'''
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

ROOT = os.getcwd()
SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok = True)
CARDS = [re.sub('\n', '', l) for l in open(f'{ROOT}/data/master_lists/master_card_list.txt')]
RELICS = [re.sub('\n', '', l) for l in open(f'{ROOT}/data/master_lists/master_relic_list.txt')]
N_CARDS, N_REL = len(CARDS), len(RELICS)
FLOOR = N_CARDS + N_REL
meta = json.load(open(f'{SRC}/meta.json'))
MC = {c: i for i, c in enumerate(meta['M_COLS'])}
FILES = sorted(int(re.search(r'X_(\d+)', f).group(1)) for f in glob.glob(f'{SRC}/X_*.npy'))
K = len(FILES)

up = [(i, CARDS.index(n[:-2]), n[:-2]) for i, n in enumerate(CARDS) if n.endswith('_u') and n[:-2] in CARDS and not n.startswith('searingblow')]
up_idx, base_idx = np.array([u for u, _, _ in up]), np.array([b for _, b, _ in up])
names = [n for _, _, n in up]
up_cols = np.array([i for i, n in enumerate(CARDS) if n.endswith('_u')])
n_up = len(up)

def derived(X, M):
    # Candidate confounders per row
    deck = X[:, :N_CARDS].sum(1).astype(float)
    nup = X[:, up_cols].sum(1).astype(float)
    rel = (X[:, N_CARDS:N_CARDS + N_REL] > 0).sum(1).astype(float)
    floor = X[:, FLOOR].astype(int)
    char = np.argmax(X[:, FLOOR + 1:FLOOR + 5], 1)
    exp = M[:, MC['player_experience']]
    # A few runs have modded values (max hp 99,999, gold 1.6M), clipped so they do not dominate averages
    hp = np.clip(M[:, MC['hp']], 0, 200) / np.clip(M[:, MC['max_hp']], 1, 200)
    hp[np.isnan(M[:, MC['hp']]) | np.isnan(M[:, MC['max_hp']])] = np.nan
    hp = np.minimum(hp, 1)
    gold = np.clip(M[:, MC['gold']], 0, 3000)
    share = M[:, MC['n_smith_before']] / np.maximum(nup, 1)
    return deck, nup, rel, floor, char, exp, hp, gold, share

# Pass 0: bin edges from train rows of the metadata only
ex, go = [], []
for k in FILES:
    M = np.load(f'{SRC}/M_{k}.npy')
    t = M[:, MC['is_test']] == 0
    ex.append(M[t, MC['player_experience']][::20])
    go.append(M[t, MC['gold']][::20])
ex, go = np.concatenate(ex), np.concatenate(go)
EXP_EDGES = np.nanquantile(ex, [.2, .4, .6, .8])
GOLD_EDGES = np.nanquantile(go, [.25, .5, .75])
HP_EDGES = np.array([.4, .7, .9])

def binned(v, edges):
    b = np.digitize(np.nan_to_num(v, nan = 0), edges) + 1
    b[np.isnan(v)] = 0
    return b

def smith_bin(share, nup):
    b = np.where(nup == 0, 0, np.where(share <= 0, 1, np.where(share <= .5, 2, np.where(share < 1, 3, 4))))
    return b

def strata(X, M, variant):
    deck, nup, rel, floor, char, exp, hp, gold, share = derived(X, M)
    fc = (np.minimum(floor, 55) // 5) * 4 + char
    if variant == 'floor_char':
        return fc, 48
    fine = ((fc * 5 + np.minimum(deck // 6, 4).astype(int)) * 4 + np.minimum(nup // 3, 3).astype(int)) * 4 + np.minimum(rel // 3, 3).astype(int)
    nfine = 48 * 80
    e, h, g, s = binned(exp, EXP_EDGES), binned(hp, HP_EDGES), binned(gold, GOLD_EDGES), smith_bin(share, nup)
    if variant == 'fine':
        return fine, nfine
    if variant == 'fine+exp':
        return fine * 6 + e, nfine * 6
    if variant == 'fine+hp':
        return fine * 5 + h, nfine * 5
    if variant == 'fine+gold':
        return fine * 5 + g, nfine * 5
    if variant == 'fine+upgrade_source':
        return fine * 5 + s, nfine * 5
    if variant == 'floor_char+all4':
        return (((fc * 6 + e) * 5 + h) * 5 + g) * 5 + s, 48 * 750
    raise ValueError(variant)

VARIANTS = ['floor_char', 'fine', 'fine+exp', 'fine+hp', 'fine+gold', 'fine+upgrade_source', 'floor_char+all4']
res = {v: np.zeros((4, strata(np.zeros((1, FLOOR + 10), np.int8), np.full((1, len(MC)), np.nan), v)[1], n_up)) for v in VARIANTS}

# Candidate means for upgraded vs plain rows: [group U, P] x [n, log_exp, hp_frac, gold, smith_share, n_rest_before, max_hp] with missing-aware counts
CAND = ['log_exp', 'hp_frac', 'gold', 'smith_share', 'rest_before', 'max_hp']
cs = np.zeros((2, len(CAND), n_up))
cn = np.zeros((2, len(CAND), n_up))
n_rows = np.zeros((2, n_up))

rng = np.random.default_rng(0)
samples_tr, samples_te = [], []

for n_done, k in enumerate(FILES):
    X, M = np.load(f'{SRC}/X_{k}.npy', mmap_mode = 'r'), np.load(f'{SRC}/M_{k}.npy')
    for s in range(0, len(X), 200000):
        x = np.asarray(X[s:s + 200000])
        m = M[s:s + 200000]
        test = m[:, MC['is_test']] == 1
        deck, nup, rel, floor, char, exp, hp, gold, share = derived(x, m)
        y = (x[:, -1] > 0).astype(float)
        feats = np.column_stack([floor, char, np.log1p(exp), hp, gold, share, m[:, MC['n_rest_before']], np.clip(m[:, MC['max_hp']], 1, 200), deck, nup, rel, y])
        keep = rng.random(len(x)) < (0.12 if True else 1)
        samples_tr.append(feats[~test & keep])
        samples_te.append(feats[test & keep])
        xt, mt = x[~test], m[~test]
        if len(xt) == 0:
            continue
        yt = y[~test]
        plain = xt[:, base_idx] > 0
        upg = xt[:, up_idx] > 0
        P = (plain & ~upg).astype(float)
        U = upg.astype(float)
        for v in VARIANTS:
            sid, ns = strata(xt, mt, v)
            S = sparse.csr_matrix((np.ones(len(xt)), (sid, np.arange(len(xt)))), shape = (ns, len(xt)))
            res[v][0] += S @ (U * yt[:, None])
            res[v][1] += S @ (U * (1 - yt)[:, None])
            res[v][2] += S @ (P * yt[:, None])
            res[v][3] += S @ (P * (1 - yt)[:, None])
        _, _, _, _, _, exp_t, hp_t, gold_t, share_t = derived(xt, mt)
        vals = [np.log1p(exp_t), hp_t, gold_t, share_t, mt[:, MC['n_rest_before']], np.clip(mt[:, MC['max_hp']], 1, 200)]
        for g, G in enumerate((U, P)):
            n_rows[g] += G.sum(0)
            for c, vv in enumerate(vals):
                ok = ~np.isnan(vv)
                cs[g, c] += G[ok].T @ vv[ok]
                cn[g, c] += G[ok].sum(0)
    print('file', n_done + 1, '/', K, flush = True)

def mh(r):
    a, b, c, d = r
    n = a + b + c + d
    ok = (a + b > 0) & (c + d > 0)
    num = np.where(ok, a * d / np.maximum(n, 1), 0).sum(0)
    den = np.where(ok, b * c / np.maximum(n, 1), 0).sum(0)
    return np.log(np.maximum(num, 1e-9) / np.maximum(den, 1e-9)), (np.where(ok, a + c, 0)).sum(0)

tab = pd.DataFrame({'card': names, 'n_U': n_rows[0], 'n_P': n_rows[1]})
for v in VARIANTS:
    tab[f'mh_{v}'] = mh(res[v])[0]
for c, nm in enumerate(CAND):
    tab[f'{nm}_U'] = cs[0, c] / np.maximum(cn[0, c], 1)
    tab[f'{nm}_P'] = cs[1, c] / np.maximum(cn[1, c], 1)

# Cards flagged by the model (v22 finetune multi head, upgrade effect below -0.05, 1,000+ train rows in both states), joined on card name
ue = pd.read_csv(f'{ROOT}/model_objects/bench/item11_a/upgrade_effects.csv')
ue = ue[(ue.n_train_plain >= 1000) & (ue.n_train_upgraded >= 1000)]
flag = ue.groupby('card').upgrade_effect_fnn_v22_ft_mt_cum.min()
tab['model_effect'] = tab.card.map(flag)
tab['flagged'] = tab.model_effect < -0.05
tab['eligible'] = (tab.n_U >= 1000) & (tab.n_P >= 1000)
tab.to_csv(f'{OUT}/confounder_cards.csv', index = False)

# Summary: mean MH log OR per variant for eligible cards, and for the flagged ones, plus candidate gaps
e, f = tab[tab.eligible], tab[tab.eligible & tab.flagged]
rows = []
for v in VARIANTS:
    col = f'mh_{v}'
    rows.append({'variant': v, 'mean_all': e[col].mean(), 'share_neg_all': (e[col] < 0).mean(), 'mean_flagged': f[col].mean(), 'share_neg_flagged': (f[col] < 0).mean(),
                 'corr_with_model_effect': e[[col, 'model_effect']].dropna().corr().iloc[0, 1]})
pd.DataFrame(rows).to_csv(f'{OUT}/summary_mh.csv', index = False)
print(pd.DataFrame(rows).round(3).to_string(index = False))
print(f'eligible {len(e)} flagged {len(f)}')

gap = []
for nm in CAND:
    d = e[f'{nm}_U'] - e[f'{nm}_P']
    gap.append({'candidate': nm, 'mean_gap_all': d.mean(), 'mean_gap_flagged': d[e.flagged].mean(), 'mean_level_P': e[f'{nm}_P'].mean()})
pd.DataFrame(gap).to_csv(f'{OUT}/summary_gaps.csv', index = False)
print(pd.DataFrame(gap).round(3).to_string(index = False))

# Predictiveness of each candidate: logistic baselines on a row sample (train fit, test evaluation)
tr, te = np.concatenate(samples_tr), np.concatenate(samples_te)
def design(a, cols):
    floor, char = a[:, 0].astype(int), a[:, 1].astype(int)
    fb = np.minimum(floor, 55) // 3
    base = np.zeros((len(a), 20 * 4))
    base[np.arange(len(a)), fb * 4 + char] = 1
    parts = [base]
    for c in cols:
        if c == 'log_exp':
            v = a[:, 2]; miss = np.isnan(v); parts += [np.nan_to_num(v - 13, nan = 0)[:, None], miss[:, None]]
        elif c == 'hp_frac':
            v = a[:, 3]; miss = np.isnan(v); parts += [np.nan_to_num(v, nan = 0)[:, None], miss[:, None]]
        elif c == 'gold':
            v = a[:, 4]; miss = np.isnan(v); parts += [np.log1p(np.nan_to_num(v, nan = 0))[:, None], miss[:, None]]
        elif c == 'upgrade_source':
            parts += [np.nan_to_num(a[:, 5], nan = 0)[:, None], a[:, 9:10] / 10]
        elif c == 'rest_before':
            parts += [a[:, 6:7]]
        elif c == 'max_hp':
            parts += [np.nan_to_num(a[:, 7:8], nan = 0) / 100]
        elif c == 'deck':
            parts += [a[:, 8:9] / 20, a[:, 9:10] / 10, a[:, 10:11] / 10]
    return np.hstack(parts)

base_res = []
for label, cols in [('floor x character', []), ('+ log experience', ['log_exp']), ('+ hp fraction', ['hp_frac']), ('+ gold', ['gold']), ('+ upgrade source', ['upgrade_source']),
                    ('+ rests before', ['rest_before']), ('+ max hp', ['max_hp']), ('+ deck size/upgrades/relics', ['deck']),
                    ('+ deck + experience', ['deck', 'log_exp']), ('+ deck + hp + gold', ['deck', 'hp_frac', 'gold']),
                    ('+ deck + exp + hp + gold + source', ['deck', 'log_exp', 'hp_frac', 'gold', 'upgrade_source', 'rest_before', 'max_hp'])]:
    clf = LogisticRegression(C = 1.0, max_iter = 300).fit(design(tr, cols), tr[:, -1])
    p = clf.predict_proba(design(te, cols))[:, 1]
    base_res.append({'model': label, 'test_log_loss': log_loss(te[:, -1], p), 'test_auc': roc_auc_score(te[:, -1], p)})
    print(base_res[-1], flush = True)
pd.DataFrame(base_res).to_csv(f'{OUT}/summary_baselines.csv', index = False)
print('rows used: train', len(tr), 'test', len(te))
