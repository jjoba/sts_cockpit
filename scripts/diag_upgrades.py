'''
Item 11, step D: do rows with the upgraded copy of a card differ from rows with only the plain copy, and does the raw data (no model) show a negative upgrade association?
Run from the project root. Streams data/training_data/finetuning/train/train.bin (Heart label = last column).
Output: diag_upgrades.csv in the scratchpad.
'''
import sys
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = '/Users/johnjoba/local_git_repos/sts_cockpit'
OUT = sys.argv[1]
CARDS = [l.strip() for l in open(f'{ROOT}/data/master_lists/master_card_list.txt')]
RELICS = [l.strip() for l in open(f'{ROOT}/data/master_lists/master_relic_list.txt')]
NAMES = CARDS + RELICS + ['floor', 'defect', 'ironclad', 'the_silent', 'watcher']
IDX = {n: i for i, n in enumerate(NAMES)}
NF, NC = len(NAMES), len(NAMES) + 5
FLOOR, N_CARDS, N_REL = IDX['floor'], len(CARDS), len(RELICS)

up = [(IDX[n[:-2]], i) for i, n in enumerate(CARDS) if n.endswith('_u') and n[:-2] in IDX and not n.startswith('searingblow')]
base_idx, up_idx = np.array([b for b, _ in up]), np.array([u for _, u in up])
up_cols = np.array([i for i, n in enumerate(CARDS) if n.endswith('_u')])
card_cols = np.arange(N_CARDS)
print(len(up), 'cards with upgraded column')

d = np.memmap(f'{ROOT}/data/training_data/finetuning/train/train.bin', dtype = np.int8, mode = 'r').reshape(-1, NC)
n_rows = len(d)

# strata A: floor bin x character. strata B adds deck size bin, number of upgrades in the deck bin, relic count bin
def strata(x, fine):
    floor = x[:, FLOOR].astype(int)
    char = np.argmax(x[:, FLOOR + 1:FLOOR + 5], 1)
    s = (np.minimum(floor, 55) // 5) * 4 + char
    if not fine:
        return s, 12 * 4
    deck = x[:, card_cols].sum(1).astype(int)
    nup = x[:, up_cols].sum(1).astype(int)
    rel = (x[:, N_CARDS:N_CARDS + N_REL] > 0).sum(1)
    s = s * 5 + np.minimum(deck // 6, 4)
    s = s * 4 + np.minimum(nup // 3, 3)
    s = s * 4 + np.minimum(rel // 3, 3)
    return s, 12 * 4 * 5 * 4 * 4

n_up = len(up)
res = {}
for fine in (False, True):
    ns = strata(np.zeros((1, NC), dtype = np.int8), fine)[1] if False else (48 if not fine else 48 * 80)
    res[fine] = np.zeros((4, ns, n_up))  # [U win, U loss, P win, P loss]
sums = np.zeros((2, 5, n_up))  # group (U, P) x [n, floor, deck, nup, rel]

chunk = 150000
for start in range(0, n_rows, chunk):
    x = np.asarray(d[start:start + chunk])
    y = (x[:, -1] > 0).astype(float)
    plain = x[:, base_idx] > 0
    upg = x[:, up_idx] > 0
    P = (plain & ~upg).astype(float)
    U = upg.astype(float)
    floor = x[:, FLOOR].astype(float)
    deck = x[:, card_cols].sum(1).astype(float)
    nup = x[:, up_cols].sum(1).astype(float)
    rel = (x[:, N_CARDS:N_CARDS + N_REL] > 0).sum(1).astype(float)
    for g, M in enumerate((U, P)):
        for k, v in enumerate((np.ones(len(x)), floor, deck, nup, rel)):
            sums[g, k] += M.T @ v
    for fine in (False, True):
        s, ns = strata(x, fine)
        S = sparse.csr_matrix((np.ones(len(x)), (s, np.arange(len(x)))), shape = (res[fine].shape[1], len(x)))
        res[fine][0] += S @ (U * y[:, None])
        res[fine][1] += S @ (U * (1 - y)[:, None])
        res[fine][2] += S @ (P * y[:, None])
        res[fine][3] += S @ (P * (1 - y)[:, None])
    print(start, end = '\r')

def mh_log_or(r):
    a, b, c, dd = r  # U win, U loss, P win, P loss  (strata x cards)
    n = a + b + c + dd
    ok = (a + b > 0) & (c + dd > 0)
    num = np.where(ok, a * dd / np.maximum(n, 1), 0).sum(0)
    den = np.where(ok, b * c / np.maximum(n, 1), 0).sum(0)
    return np.log(np.maximum(num, 1e-9) / np.maximum(den, 1e-9)), (a + b)[ok].sum(0) if False else None

rows = pd.DataFrame({'card': [CARDS[b] for b in base_idx]})
rows['n_U'], rows['n_P'] = sums[0, 0], sums[1, 0]
for k, nm in enumerate(['floor', 'deck', 'n_upgrades', 'relics'], 1):
    rows[f'{nm}_U'] = sums[0, k] / sums[0, 0]
    rows[f'{nm}_P'] = sums[1, k] / sums[1, 0]
rows['winrate_U'] = res[False][0].sum(0) / (res[False][0] + res[False][1]).sum(0)
rows['winrate_P'] = res[False][2].sum(0) / (res[False][2] + res[False][3]).sum(0)
rows['mh_logor_floor_char'] = mh_log_or(res[False])[0]
rows['mh_logor_fine'] = mh_log_or(res[True])[0]
rows.to_csv(OUT, index = False)
print('\nwrote', OUT)
