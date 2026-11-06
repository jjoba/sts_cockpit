'''
Test bench for how well the models capture card and relic interactions and upgrade states. Run from the project root:

    python scripts/interaction_bench.py models=fnn_v21_ft_st,fnn_v22_ft_st_cum name=v22_cum_vs_split

Models are TorchScript files in model_objects/production, the Heart head (last output) is the one tested. Options (key=value):
    models    comma separated model names (required)
    name      output folder under model_objects/bench (default: first model name)
    sections  combos, upgrades, picks (default these three), agreement (opt in; needs two or more models; also uses data/bench/decisions_v24_relic_*.npy from extract_relic_decisions.py)
    data      data folder under data/ (default training_data). Folders written by process_data.py since v24 carry HP / max HP / gold columns after the
              929 features (layout.json); models are fed the extras their config.json lists, models without extras ignore them. The counterfactual
              contexts keep each held-out deck's own HP and gold. The picks section then reads decisions_v24_X.npy / _M.npy (HP and gold of the reward's floor)
    n_null    random control pairs per character (default 25)
    seed      seed for control pairs, contexts and the control row sample (default 0)

What each section answers
  combos    Does the model give the groups in interaction_combos.py extra credit, and is anything left over?
            model_interaction: counterfactual. Add the cards to held-out decks that have none of them and take the mixed difference of the logit
                               (0 = purely additive in those cards).
            residual_*:        logistic regression of the real outcome on card presence with the model's logit as offset, the coefficient of "all cards
                               present" is the log-odds the model leaves unexplained (> 0 = the model undervalues the combo). Run clustered (rows of a
                               run are adjacent in the files) so the z scores are not inflated by the ~5 rows each run contributes.
            The null controls are random pairs of common cards with no link; their residuals show how big noise is for pairs that should show nothing.
  upgrades  Is what is learned about a card shared by its upgraded copy?
            cells:    the pair interaction computed separately for plain / upgraded versions of each card (a model that shares across upgrade
                      states gives similar values), with the training rows each version combination has.
            effects:  model-implied logit change from upgrading each card.
            rare:     held-out calibration of the plain / upgraded copy of every card, grouped by how many training rows had that state.
  picks     Card pick decisions of A20 held-out runs (data/bench/decisions*_X.npy / _M.npy, built by extract_pick_decisions.py):
            per-card calibration after the pick, and how often the model's best option is the one the player took.
'''

import json
import os
import sys
import itertools
from time import time

import numpy as np
import pandas as pd
import torch
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from interaction_combos import COMBOS, NULL_EXCLUDE

ROOT = os.getcwd()
CARDS = [l.strip() for l in open(f'{ROOT}/data/master_lists/master_card_list.txt')]
RELICS = [l.strip() for l in open(f'{ROOT}/data/master_lists/master_relic_list.txt')]
NAMES = CARDS + RELICS + ['floor', 'defect', 'ironclad', 'the_silent', 'watcher']
IDX = {n: i for i, n in enumerate(NAMES)}
N_FEATURES, N_HEADS = len(NAMES), 5
N_COLS = N_FEATURES + N_HEADS
DATA_NAME, N_EXTRA, EXTRA_COLS = 'training_data', 0, []
EXTRA_GROUPS = {'hp': ['hp_pct', 'max_hp'], 'gold': ['gold_25']}
FLOOR = IDX['floor']
CHARS = ['defect', 'ironclad', 'the_silent', 'watcher']
CHAR_COL = {c: IDX[c] for c in CHARS}
MIN_FLOOR = 20 # combos are judged on decks from this floor on

def configure(data_name):
    # Row layout of the data folder: features, then (v24 data) the extra run state columns, then the head labels
    global DATA_NAME, N_COLS, N_EXTRA, EXTRA_COLS
    DATA_NAME = data_name
    layout = f'{ROOT}/data/{data_name}/layout.json'
    EXTRA_COLS = json.load(open(layout))['extra_cols'] if os.path.exists(layout) else []
    N_EXTRA = len(EXTRA_COLS)
    N_COLS = N_FEATURES + N_EXTRA + N_HEADS

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
sigmoid = lambda z: 1 / (1 + np.exp(-z))

############### MODELS ##############

class Net:
    # Forward pass of a saved model, taking the Heart head's logit. Cumulative upgrade models carry base_idx / up_idx buffers
    def __init__(self, name):
        self.name = name
        self.path = f'{ROOT}/model_objects/production/{name}.pth'
        sd = torch.jit.load(self.path, map_location = 'cpu').state_dict()
        self.base_idx = sd['base_idx'].to(device) if 'base_idx' in sd else None
        self.up_idx = sd['up_idx'].to(device) if 'up_idx' in sd else None
        cfg_path = f'{ROOT}/model_objects/dev/{name}/config.json'
        extras = json.load(open(cfg_path))['config'].get('extra_features', []) if os.path.exists(cfg_path) else []
        self.extras = '+'.join(extras)
        self.extra_names = [c for g in extras for c in EXTRA_GROUPS[g]]
        self.scale = sd['input_scale'].to(device) if 'input_scale' in sd else None
        assert len(self.extra_names) == (0 if self.scale is None else len(self.scale) - N_FEATURES), f'{name}: extra inputs do not match its saved input_scale'
        self.cols = None # columns of a data row fed to the network, set by use_layout()
        sd = {k: v for k, v in sd.items() if k.startswith('linear_relu_stack')}
        keys = sorted({k.rsplit('.', 1)[0] for k in sd}, key = lambda s: int(s.split('.')[-1]))
        self.layers = [(sd[k + '.weight'].to(device), sd[k + '.bias'].to(device)) for k in keys]

    def use_layout(self):
        missing = [c for c in self.extra_names if c not in EXTRA_COLS]
        if missing:
            raise SystemExit(f'{self.name} needs {missing}, data/{DATA_NAME} does not have them')
        self.cols = torch.tensor(list(range(N_FEATURES)) + [N_FEATURES + EXTRA_COLS.index(c) for c in self.extra_names], device = device)

    def tag(self):
        st = os.stat(self.path)
        return f'{self.name}_{st.st_size}_{int(st.st_mtime)}'

    @torch.no_grad()
    def logit(self, X, batch = 100000):
        out = []
        for i in range(0, len(X), batch):
            x = torch.as_tensor(np.asarray(X[i:i + batch]), dtype = torch.float32).to(device)
            if self.cols is None:
                self.use_layout()
            x = x.index_select(1, self.cols)
            if self.scale is not None:
                x = x * self.scale
            if self.base_idx is not None:
                x = x.index_add(1, self.base_idx, x.index_select(1, self.up_idx))
            for j, (W, b) in enumerate(self.layers):
                x = x @ W.T + b
                if j < len(self.layers) - 1:
                    x = torch.relu(x)
            out.append(x[:, -1].float().cpu())
        return torch.cat(out).numpy().astype(np.float64)

############### DATA ##############

def open_split(split):
    return np.memmap(f'{ROOT}/data/{DATA_NAME}/finetuning/{split}/{split}.bin', dtype = np.int8, mode = 'r').reshape(-1, N_COLS)

def base_name(n):
    return n[:-2] if n.endswith('_u') and n[:-2] in IDX else n

def item_cols(item):
    # (plain column, upgraded column or None)
    return IDX[item], IDX.get(item + '_u')

def common_cards(n_rows = 1500000):
    # Per character: fraction of its decks (floor >= MIN_FLOOR) holding each column, from the first rows of the training file. Also each column's usual character
    d = open_split('train')
    r = np.asarray(d[:n_rows])
    r = r[r[:, FLOOR] >= MIN_FLOOR]
    held = r[:, :N_FEATURES] > 0
    freq = {c: held[r[:, CHAR_COL[c]] == 1].mean(0) for c in CHARS}
    owner = np.argmax([freq[c] for c in CHARS], 0)
    return freq, owner

def choose_null_pairs(freq, n_null, seed):
    # Random pairs from each character's most common non-starter cards, none of which are a listed combo
    rng = np.random.default_rng(seed)
    listed = {c: {frozenset(items) for items, _ in COMBOS[c]} for c in COMBOS}
    pairs = {}
    for c in CHARS:
        cand = [i for i in np.argsort(-freq[c]) if i < len(CARDS) and not NAMES[i].endswith('_u') and NAMES[i] not in NULL_EXCLUDE
                and NAMES[i] + '_u' in IDX][:45]
        names = [NAMES[i] for i in cand]
        got = set()
        while len(got) < n_null:
            a, b = rng.choice(len(names), 2, replace = False)
            p = frozenset((names[a], names[b]))
            if p not in listed[c]:
                got.add(p)
        pairs[c] = [tuple(sorted(p)) for p in sorted(got, key = sorted)]
    return pairs

def build_combo_table(null_pairs):
    rows = []
    for c in CHARS:
        for items, why in COMBOS[c]:
            missing = [i for i in items if i not in IDX]
            if missing:
                print(f'  skipping {items}: not in the master lists ({missing})')
                continue
            rows.append({'char': c, 'items': items, 'kind': 'synergy', 'why': why})
        for items in null_pairs[c]:
            rows.append({'char': c, 'items': items, 'kind': 'null', 'why': 'random common pair'})
    return rows

def scan(split, nets, combos, cache_dir, support_and_cells, pool_rate, rng):
    '''
    One pass over a finetuning file. Returns per row metadata and each model's Heart logit (cached per model file), presence (plain / upgraded) of every item
    used in a combo for rows at MIN_FLOOR+, and for the test file also a random pool of decks to use as counterfactual contexts and the card state cells
    '''
    d = open_split(split)
    n = len(d)
    items = sorted({i for c in combos for i in c['items']}, key = lambda s: IDX[s])
    plain_cols = np.array([IDX[i] for i in items])
    up_cols = np.array([IDX.get(i + '_u', 0) for i in items]) # items without an upgraded column are masked by has_up
    has_up = np.array([(i + '_u') in IDX for i in items])

    cache = {m.name: f'{cache_dir}/{split}_{m.tag()}.npy' for m in nets}
    todo = [m for m in nets if not os.path.exists(cache[m.name])]
    logits = {m.name: np.zeros(n, np.float32) for m in todo}

    y = np.zeros(n, np.int8); floor = np.zeros(n, np.int16); char = np.zeros(n, np.int8); cid = np.zeros(n, np.int64)
    keep_rows, P, U, pool, pool_char = [], [], [], [], []
    cells, cell_support = [], None
    upg = [i for i in range(len(CARDS)) if NAMES[i] + '_u' in IDX and not NAMES[i].startswith('searingblow')]
    upg_plain, upg_up = np.array(upg), np.array([IDX[NAMES[i] + '_u'] for i in upg])
    if support_and_cells and split == 'train':
        cell_support = np.zeros((2, len(upg)), np.int64)

    last_floor, next_cid, t0 = None, 0, time()
    step = 400000
    for lo in range(0, n, step):
        r = np.asarray(d[lo:lo + step])
        X = r[:, :N_FEATURES + N_EXTRA] # features and the extra run state, the labels are not fed to the models
        f = X[:, FLOOR].astype(np.int16)
        # Rows of one run are adjacent with rising floors, a drop in floor starts the next run
        drop = np.zeros(len(r), bool)
        drop[1:] = f[1:] < f[:-1]
        drop[0] = last_floor is not None and f[0] < last_floor
        c_ = next_cid + np.cumsum(drop)
        next_cid, last_floor = int(c_[-1]), int(f[-1])
        y[lo:lo + len(r)] = r[:, -1]; floor[lo:lo + len(r)] = f; cid[lo:lo + len(r)] = c_
        char[lo:lo + len(r)] = X[:, 925:929].argmax(1)
        for m in todo:
            logits[m.name][lo:lo + len(r)] = m.logit(X)

        k = f >= MIN_FLOOR
        keep_rows.append(np.flatnonzero(k) + lo)
        P.append(X[k][:, plain_cols] > 0)
        U.append((X[k][:, up_cols] > 0) & has_up)
        if support_and_cells:
            hp, hu = X[:, upg_plain] > 0, X[:, upg_up] > 0
            if split == 'train':
                cell_support[0] += hp.sum(0); cell_support[1] += hu.sum(0)
            else:
                cells.append(sparse.csr_matrix(np.concatenate([hp, hu], 1)))
        if pool_rate:
            sel = (f >= 15) & (f <= 45) & (rng.random(len(r)) < pool_rate)
            pool.append(X[sel].copy()); pool_char.append(X[sel][:, 925:929].argmax(1))
        print(f'  {split} rows {lo + len(r):,}/{n:,}  {time() - t0:.0f}s', end = '\r', flush = True)
    print()

    for m in todo:
        np.save(cache[m.name], logits[m.name])
    z = {m.name: np.load(cache[m.name]) for m in nets}

    keep = np.concatenate(keep_rows)
    out = {'n': n, 'y': y, 'floor': floor, 'char': char, 'cid': cid, 'z': z, 'items': items, 'keep': keep,
           'P': np.concatenate(P), 'U': np.concatenate(U), 'upg_cards': upg, 'cell_support': cell_support}
    if cells:
        out['cells'] = sparse.vstack(cells).tocsr()
    if pool:
        out['pool'] = np.concatenate(pool); out['pool_char'] = np.concatenate(pool_char)
    return out

############### COMBOS ##############

def irls(y, D, offset, w, cid, iters = 10):
    # Weighted logistic regression with a fixed offset, with run-clustered standard errors (cid is sorted)
    k = D.shape[1]
    b = np.zeros(k)
    for _ in range(iters):
        p = sigmoid(offset + D @ b)
        H = D.T @ (D * (w * p * (1 - p))[:, None]) + 1e-6 * np.eye(k)
        step = np.linalg.solve(H, D.T @ (w * (y - p)))
        b += step
        if np.abs(step).max() < 1e-6:
            break
    p = sigmoid(offset + D @ b)
    H = D.T @ (D * (w * p * (1 - p))[:, None]) + 1e-6 * np.eye(k)
    starts = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1]])
    S = np.add.reduceat(D * (w * (y - p))[:, None], starts, axis = 0)
    Hi = np.linalg.inv(H)
    V = Hi @ (S.T @ S) @ Hi
    return b, np.sqrt(np.diag(V))

def cluster_z(res, cid):
    # z score of a sum of residuals with run-clustered variance
    starts = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1]])
    return res.sum() / np.sqrt((np.add.reduceat(res, starts) ** 2).sum())

def mixed_difference(net, ctx, cols):
    # Interaction of adding one copy of each column: inclusion-exclusion of the logit over all subsets, per context row
    total = 0
    for s in range(len(cols) + 1):
        for sub in itertools.combinations(cols, s):
            x = ctx.copy()
            if sub:
                x[:, list(sub)] += 1
            total = total + (-1) ** (len(cols) - s) * net.logit(x)
    return total

def contexts_for(pool, pool_char, char, items, n, rng):
    # Held-out decks of this character, floors 15-45, none of which holds any of the items
    ok = pool_char == CHARS.index(char)
    for i in items:
        p, u = item_cols(i)
        ok &= pool[:, p] == 0
        if u is not None:
            ok &= pool[:, u] == 0
    idx = np.flatnonzero(ok)
    if len(idx) > n:
        idx = rng.choice(idx, n, replace = False)
    return pool[idx]

def run_combos(nets, combos, data, rng):
    items_idx = {i: k for k, i in enumerate(data['train']['items'])}
    rows = []
    sets = {}
    for split in ['train', 'test']:
        D = data[split]
        keep = D['keep']
        sets[split] = {'char': D['char'][keep], 'y': D['y'][keep].astype(float), 'cid': D['cid'][keep], 'z': {m: D['z'][m][keep].astype(np.float64) for m in D['z']},
                       'pres': D['P'] | D['U'], 'P': D['P'], 'U': D['U']}
        # The share of rows with none of a combo's cards is large, a fixed random subset of them (weighted back up) keeps the fits fast
        sets[split]['u'] = np.random.default_rng(7).random(len(keep))

    pool, pool_char = data['test']['pool'], data['test']['pool_char']
    for k, combo in enumerate(combos):
        c, items = combo['char'], combo['items']
        cols = [items_idx[i] for i in items]
        row = {'char': c, 'combo': ' + '.join(items), 'size': len(items), 'kind': combo['kind'], 'why': combo['why']}
        ctx = contexts_for(pool, pool_char, c, items, 2000, rng)
        row['n_context'] = len(ctx)

        for split, S in sets.items():
            in_char = S['char'] == CHARS.index(c)
            pres = S['pres'][:, cols]
            all_present = in_char & pres.all(1)
            any_present = in_char & pres.any(1)
            none = in_char & ~pres.any(1)
            rate = min(1.0, 150000 / max(none.sum(), 1))
            use = any_present | (none & (S['u'] < rate))
            w = np.where(any_present[use], 1.0, 1 / rate)
            Z = pres[use].astype(float)
            subsets = [s for r_ in range(1, len(items) + 1) for s in itertools.combinations(range(len(items)), r_)]
            D = np.stack([np.ones(len(Z))] + [Z[:, list(s)].prod(1) for s in subsets], 1)
            full = len(subsets)
            row[f'n_{split}'] = int(all_present.sum())
            row[f'wins_{split}'] = int(S['y'][all_present].sum())
            cid_u, y_u = S['cid'][use], S['y'][use]
            for m in nets:
                z = S['z'][m.name]
                if all_present.sum() >= 20:
                    b, se = irls(y_u, D, z[use], w, cid_u)
                    row[f'residual_{split}_{m.name}'] = b[full]
                    row[f'residual_se_{split}_{m.name}'] = se[full]
                    p = sigmoid(z[all_present])
                    row[f'oe_{split}_{m.name}'] = S['y'][all_present].sum() / p.sum()
                    row[f'oe_z_{split}_{m.name}'] = cluster_z(S['y'][all_present] - p, S['cid'][all_present])
        if len(ctx) >= 100:
            xcols = [IDX[i] for i in items]
            for m in nets:
                row[f'model_interaction_{m.name}'] = float(mixed_difference(m, ctx, xcols).mean())
        rows.append(row)
        print(f'  combos {k + 1}/{len(combos)}', end = '\r', flush = True)
    print()
    return pd.DataFrame(rows)

def summarise_combos(df, nets):
    out = []
    for m in nets:
        row = {'model': m.name}
        for kind in ['synergy', 'null']:
            d = df[df.kind == kind]
            inter = d[f'model_interaction_{m.name}'].dropna()
            row[f'{kind}_n'] = len(d)
            row[f'{kind}_model_interaction'] = inter.mean()
            for split in ['train', 'test']:
                r = d[f'residual_{split}_{m.name}'].dropna()
                z = (r / d.loc[r.index, f'residual_se_{split}_{m.name}'])
                row[f'{kind}_residual_{split}'] = r.mean()
                row[f'{kind}_abs_residual_{split}'] = r.abs().mean()
                row[f'{kind}_share_z_gt2_{split}'] = (z > 2).mean()
                row[f'{kind}_share_z_lt_m2_{split}'] = (z < -2).mean()
        out.append(row)
    return pd.DataFrame(out)

############### UPGRADES ##############

def run_upgrade_cells(nets, combos, data, rng):
    # Pair interaction by upgrade state of each card, with the training rows each combination of states has
    items_idx = {i: k for k, i in enumerate(data['train']['items'])}
    D = data['train']
    k_ = D['keep']
    pool, pool_char = data['test']['pool'], data['test']['pool_char']
    rows = []
    for combo in combos:
        items, c = combo['items'], combo['char']
        if len(items) != 2 or any((i + '_u') not in IDX for i in items):
            continue
        a, b = items
        ctx = contexts_for(pool, pool_char, c, items, 2000, rng)
        if len(ctx) < 100:
            continue
        ia, ib = items_idx[a], items_idx[b]
        in_char = D['char'][k_] == CHARS.index(c)
        for sa, sb in itertools.product(['plain', 'upgraded'], repeat = 2):
            col_a = IDX[a] if sa == 'plain' else IDX[a + '_u']
            col_b = IDX[b] if sb == 'plain' else IDX[b + '_u']
            has_a = (D['P'][:, ia] if sa == 'plain' else D['U'][:, ia])
            has_b = (D['P'][:, ib] if sb == 'plain' else D['U'][:, ib])
            row = {'char': c, 'combo': f'{a} + {b}', 'kind': combo['kind'], 'state_a': sa, 'state_b': sb, 'n_train': int((in_char & has_a & has_b).sum())}
            for m in nets:
                row[f'model_interaction_{m.name}'] = float(mixed_difference(m, ctx, [col_a, col_b]).mean())
            rows.append(row)
    return pd.DataFrame(rows)

def summarise_cells(cells, nets):
    out = []
    for m in nets:
        col = f'model_interaction_{m.name}'
        spreads, rare_gap = [], []
        for _, g in cells.groupby('combo'):
            spreads.append(g[col].max() - g[col].min())
            big = g.n_train.max()
            # the interaction in the rarest combination of states vs the best supported one
            if big > 0:
                rare_gap.append(abs(g.loc[g.n_train.idxmin(), col] - g.loc[g.n_train.idxmax(), col]))
        out.append({'model': m.name, 'pairs': len(spreads), 'mean_spread_across_states': np.mean(spreads), 'median_spread_across_states': np.median(spreads),
                    'mean_rarest_vs_best_supported_gap': np.mean(rare_gap)})
    return pd.DataFrame(out)

def run_upgrade_effects(nets, data, owner, rng):
    # Model-implied logit change from adding the upgraded rather than plain copy of each card, on held-out decks of the card's usual character without it
    D = data['train']
    pool, pool_char = data['test']['pool'], data['test']['pool_char']
    rows = []
    for j, card in enumerate(D['upg_cards']):
        name = NAMES[card]
        ctx = contexts_for(pool, pool_char, CHARS[owner[card]], [name], 1500, rng)
        if len(ctx) < 100:
            continue
        row = {'card': name, 'char': CHARS[owner[card]], 'n_train_plain': int(D['cell_support'][0, j]), 'n_train_upgraded': int(D['cell_support'][1, j])}
        for m in nets:
            x = ctx.copy(); x[:, card] += 1
            plain = m.logit(x)
            x = ctx.copy(); x[:, IDX[name + '_u']] += 1
            row[f'upgrade_effect_{m.name}'] = float((m.logit(x) - plain).mean())
        rows.append(row)
    return pd.DataFrame(rows)

def run_rare_cells(nets, data):
    # Held-out calibration of each card's plain / upgraded copy (all test rows), by how many training rows had that state
    T, R = data['test'], data['train']
    S = T['cells']
    n_upg = len(R['upg_cards'])
    support = np.concatenate([R['cell_support'][0], R['cell_support'][1]])
    cid, y = T['cid'], T['y'].astype(float)
    # clusters x rows indicator, to sum residuals within a run for each cell
    _, inv = np.unique(cid, return_inverse = True)
    C = sparse.csr_matrix((np.ones(len(cid)), (inv, np.arange(len(cid)))))
    rows = []
    for m in nets:
        p = sigmoid(T['z'][m.name].astype(np.float64))
        res = y - p
        O, E = S.T @ y, S.T @ p
        A = C @ S.multiply(res[:, None]).tocsr()
        var = np.asarray(A.multiply(A).sum(0)).ravel()
        n_rows = np.asarray(S.sum(0)).ravel()
        for cell in range(2 * n_upg):
            if E[cell] < 5:
                continue
            card = R['upg_cards'][cell % n_upg]
            rows.append({'model': m.name, 'card': NAMES[card], 'state': 'plain' if cell < n_upg else 'upgraded', 'n_train': int(support[cell]), 'n_test': int(n_rows[cell]),
                         'obs_wins': O[cell], 'exp_wins': E[cell], 'rel_err': O[cell] / E[cell] - 1, 'z': (O[cell] - E[cell]) / np.sqrt(var[cell]) if var[cell] > 0 else np.nan})
    return pd.DataFrame(rows)

def summarise_rare(rare):
    bins = [(0, 1000), (1000, 5000), (5000, 20000), (20000, 10 ** 12)]
    out = []
    for m, g in rare.groupby('model'):
        for lo, hi in bins:
            d = g[(g.n_train >= lo) & (g.n_train < hi)]
            if len(d) == 0:
                continue
            out.append({'model': m, 'train_rows_of_state': f'{lo}-{hi if hi < 10 ** 11 else "inf"}', 'cells': len(d), 'mean_z2': (d.z ** 2).mean(),
                        'mean_z': d.z.mean(), 'rms_rel_err': np.sqrt(np.average(d.rel_err ** 2, weights = d.exp_wins)),
                        'rel_err_pooled': d.obs_wins.sum() / d.exp_wins.sum() - 1})
    return pd.DataFrame(out)

############### CARD PICKS ##############

def run_picks(nets):
    bench = f'{ROOT}/data/bench'
    tag = '_v24' if N_EXTRA else ''
    if not os.path.exists(f'{bench}/decisions{tag}_X.npy'):
        print('  no pick decisions found (run scripts/extract_pick_decisions.py), skipping')
        return None, None
    X = np.load(f'{bench}/decisions{tag}_X.npy', mmap_mode = 'r')
    M = np.load(f'{bench}/decisions{tag}_M.npy')
    picked, offered, y, floor = M[:, 0].astype(int), M[:, 1:6].astype(int), M[:, 6].astype(int), M[:, 7].astype(int)
    N = len(M)
    has = picked >= 0
    ch = np.asarray(X[:, 925:929]).argmax(1)
    fb = np.digitize(floor, [6, 16, 31])
    grp = ch * 4 + fb
    cardname = np.array([base_name(NAMES[p]) if p >= 0 else '' for p in picked])
    uc, cnt = np.unique(cardname[has], return_counts = True)
    cards = [c for c, n in zip(uc, cnt) if n >= 1000]

    offcount, pickcount = np.zeros(len(NAMES)), np.zeros(len(NAMES))
    for k in range(5):
        v = offered[:, k]
        np.add.at(offcount, v[v >= 0], 1)
    np.add.at(pickcount, picked[has], 1)
    pop = pickcount / np.maximum(offcount, 1)

    sel = np.sort(np.random.default_rng(1).choice(N, min(N, 350000), replace = False))
    opts, pk, ys = offered[sel], picked[sel], y[sel]
    Xs = np.asarray(X[sel])
    cands = {'always skip': np.full(len(sel), -1),
             'most popular card': np.array([(o[o >= 0][np.argmax(pop[o[o >= 0]])] if (o >= 0).any() else -1) for o in opts])}

    cal_rows, agree_rows = [], []
    for m in nets:
        z0 = np.concatenate([m.logit(np.asarray(X[lo:lo + 200000])) for lo in range(0, N, 200000)])
        x1 = np.zeros(N)
        for lo in range(0, N, 200000):
            ii = np.arange(lo, min(lo + 200000, N))
            x = np.asarray(X[ii]).copy()
            h = has[ii]
            x[np.flatnonzero(h), picked[ii][h]] += 1
            x1[ii] = m.logit(x)
        p1 = sigmoid(x1)
        # Rescale to the observed win rate in each character x floor group so survivorship in who gets to a pick does not read as card error
        r = np.ones(16)
        for g in range(16):
            mk = has & (grp == g)
            if mk.any():
                r[g] = y[mk].sum() / p1[mk].sum()
        pa = np.clip(p1 * r[grp], 1e-6, 0.999)
        zs, rel, ns = [], [], []
        for c in cards:
            mk = has & (cardname == c)
            O, E, V = y[mk].sum(), pa[mk].sum(), (pa[mk] * (1 - pa[mk])).sum()
            z = (O - E) / np.sqrt(V)
            zs.append(z); rel.append(O / E - 1); ns.append(mk.sum())
            cal_rows.append({'model': m.name, 'card': c, 'picks': int(mk.sum()), 'obs_wins': int(O), 'exp_wins': E, 'z': z, 'rel_err': O / E - 1,
                             'pick_effect_logit': float(np.mean(x1[mk] - z0[mk]))})
        zs, rel, ns = np.array(zs), np.array(rel), np.array(ns)

        z_s = z0[sel]
        best, best_z = np.full(len(sel), -1), z_s.copy()
        for k in range(5):
            v = opts[:, k]
            mk = v >= 0
            if not mk.any():
                continue
            x = Xs[mk].copy()
            x[np.arange(mk.sum()), v[mk]] += 1
            zk = np.full(len(sel), -1e9); zk[mk] = m.logit(x)
            upd = zk > best_z
            best[upd] = v[upd]; best_z[upd] = zk[upd]
        cands[m.name] = best
        agree_rows.append({'model': m.name, 'decisions': int(has.sum()), 'cards_calibrated': len(cards), 'mean_z2': float(np.mean(zs ** 2)), 'share_abs_z_gt2': float(np.mean(np.abs(zs) > 2)),
                           'rms_rel_err': float(np.sqrt(np.average(rel ** 2, weights = ns)))})
    rules = pd.DataFrame([{'model': k, 'agree_all': (v == pk).mean(), 'agree_heart_killers': (v == pk)[ys == 1].mean(), 'agree_others': (v == pk)[ys == 0].mean(),
                           'picks_a_card': (v >= 0).mean()} for k, v in cands.items()])
    return rules.merge(pd.DataFrame(agree_rows), on = 'model', how = 'left'), pd.DataFrame(cal_rows)

def rank_options(m, X, opts):
    # Heart logit of the deck as it stands (column 0: the skip option) and with each offered item added (columns 1-5, -inf where the offer has fewer than 5)
    S = np.full((len(X), 6), -np.inf)
    S[:, 0] = m.logit(X)
    for k in range(5):
        v = opts[:, k]
        mk = v >= 0
        if not mk.any():
            continue
        x = X[mk].copy()
        x[np.arange(mk.sum()), v[mk]] += 1
        S[mk, k + 1] = m.logit(x)
    return S

def rank_agreement(Sa, Sb, minimum_margin = 0.1):
    # Per decision: same best option, Kendall tau between the two orderings of the available options, identical full ordering, and top-1 agreement where both models see a clear best
    ok = np.isfinite(Sa)
    n_opts = ok.sum(1)
    conc = disc = pairs = np.zeros(len(Sa))
    for a in range(6):
        for b in range(a + 1, 6):
            v = ok[:, a] & ok[:, b]
            sg = np.sign(Sa[:, a] - Sa[:, b]) * np.sign(Sb[:, a] - Sb[:, b])
            conc = conc + (v & (sg > 0)); disc = disc + (v & (sg < 0)); pairs = pairs + v
    tau = (conc - disc) / np.maximum(pairs, 1)
    top_a, top_b = Sa.argmax(1), Sb.argmax(1)

    def margin(S):
        t = np.sort(np.where(np.isfinite(S), S, -1e9), axis = 1)
        return t[:, -1] - t[:, -2]
    clear = (margin(Sa) >= minimum_margin) & (margin(Sb) >= minimum_margin)
    return {'decisions': len(Sa), 'same_top_choice': float((top_a == top_b).mean()), 'mean_kendall_tau': float(tau.mean()), 'identical_order': float((disc == 0).mean()),
            'share_clear_top_both': float(clear.mean()), 'same_top_choice_when_clear': float((top_a == top_b)[clear].mean()) if clear.any() else float('nan'),
            'skip_chosen_a': float((top_a == 0).mean()), 'skip_chosen_b': float((top_b == 0).mean())}

def run_agreement(nets):
    # How far models agree on which card (all rewards) or boss relic (relic offers) is best, as an ordering of the offered items and the option to skip, and on each item's average effect
    from scipy.stats import spearmanr
    bench = f'{ROOT}/data/bench'
    rows, per_item = [], []
    for kind, tag, n_max, min_picks in [('card', '', 350000, 1000), ('relic', '_relic', None, 100)]:
        if not os.path.exists(f'{bench}/decisions_v24{tag}_X.npy'):
            print(f'  no {kind} decisions found, skipping')
            continue
        X = np.load(f'{bench}/decisions_v24{tag}_X.npy', mmap_mode = 'r')
        M = np.load(f'{bench}/decisions_v24{tag}_M.npy')
        sel = np.arange(len(M)) if n_max is None or len(M) <= n_max else np.sort(np.random.default_rng(1).choice(len(M), n_max, replace = False))
        Xs, opts, picked = np.asarray(X[sel]), M[sel, 1:6].astype(int), M[sel, 0].astype(int)
        S = {m.name: rank_options(m, Xs, opts) for m in nets}
        base = {k: v[:, :1].copy() for k, v in S.items()}
        if kind == 'relic':
            for v in S.values():
                v[:, 0] = -np.inf # a boss relic is almost never skipped (0.8%), and the model's value of 'no relic' is not a choice worth ranking
        # Agreement with the player: share of decisions where the model's best option is the pick (0 is skip, option k is the k-th offered item)
        pick_slot = np.where(picked < 0, 0, 1 + (opts == picked[:, None]).argmax(1))
        for m in nets:
            rows.append({'kind': kind, 'a': m.name, 'b': 'player', 'decisions': len(sel), 'same_top_choice': float((S[m.name].argmax(1) == pick_slot).mean())})
        for i in range(len(nets)):
            for j in range(i + 1, len(nets)):
                r = rank_agreement(S[nets[i].name], S[nets[j].name])
                rows.append({'kind': kind, 'a': nets[i].name, 'b': nets[j].name, **r})
                # Per item effect (logit change from adding it) averaged over every time it was offered, then rank correlation across items
                eff = {}
                for m in (nets[i], nets[j]):
                    d = S[m.name][:, 1:] - base[m.name]
                    e = {}
                    for it in np.unique(opts[opts >= 0]):
                        mk = opts == it
                        if mk.sum() >= min_picks:
                            e[it] = float(d[mk].mean())
                    eff[m.name] = e
                items = sorted(set(eff[nets[i].name]) & set(eff[nets[j].name]))
                ea, eb = [eff[nets[i].name][it] for it in items], [eff[nets[j].name][it] for it in items]
                rho = spearmanr(ea, eb)[0] if len(items) > 2 else float('nan')
                rows.append({'kind': kind + '_item_effect', 'a': nets[i].name, 'b': nets[j].name, 'decisions': len(items), 'spearman': float(rho), 'pearson': float(np.corrcoef(ea, eb)[0, 1]) if len(items) > 2 else float('nan')})
                per_item += [{'kind': kind, 'item': NAMES[it], 'times_offered': int((opts == it).sum()), f'effect_{nets[i].name}': ea[k], f'effect_{nets[j].name}': eb[k]} for k, it in enumerate(items)]
    return pd.DataFrame(rows), pd.DataFrame(per_item)

############### RUN ##############

def parse(argv):
    opts = {'sections': 'combos,upgrades,picks', 'n_null': '25', 'seed': '0'}
    for a in argv:
        k, v = a.split('=', 1)
        opts[k] = v
    if 'models' not in opts:
        raise SystemExit('usage: python scripts/interaction_bench.py models=name1,name2 [name=...] [sections=combos,upgrades,picks]')
    return opts

if __name__ == '__main__':
    opts = parse(sys.argv[1:])
    names = opts['models'].split(',')
    sections = opts['sections'].split(',')
    out_dir = f"{ROOT}/model_objects/bench/{opts.get('name', names[0])}"
    configure(opts.get('data', 'training_data'))
    cache_dir = f'{ROOT}/data/bench/cache' + ('' if DATA_NAME == 'training_data' else f'_{DATA_NAME}')
    os.makedirs(out_dir, exist_ok = True); os.makedirs(cache_dir, exist_ok = True)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.float_format', lambda v: f'{v:.4f}')

    nets = [Net(n) for n in names]
    rng = np.random.default_rng(int(opts['seed']))
    summary = {}

    if 'combos' in sections or 'upgrades' in sections:
        print('Choosing control pairs')
        freq, owner = common_cards()
        combos = build_combo_table(choose_null_pairs(freq, int(opts['n_null']), int(opts['seed'])))
        print(f"{sum(c['kind'] == 'synergy' for c in combos)} listed combos, {sum(c['kind'] == 'null' for c in combos)} control pairs")
        upgrades = 'upgrades' in sections
        data = {'train': scan('train', nets, combos, cache_dir, upgrades, 0, rng), 'test': scan('test', nets, combos, cache_dir, upgrades, 0.04, rng)}

    if 'combos' in sections:
        print('Combos')
        df = run_combos(nets, combos, data, rng)
        df.to_csv(f'{out_dir}/combos.csv', index = False)
        summary['combos'] = summarise_combos(df, nets)
        summary['combos'].to_csv(f'{out_dir}/summary_combos.csv', index = False)

    if 'upgrades' in sections:
        print('Upgrades')
        cells = run_upgrade_cells(nets, combos, data, rng)
        cells.to_csv(f'{out_dir}/upgrade_cells.csv', index = False)
        summary['upgrade_cells'] = summarise_cells(cells, nets)
        effects = run_upgrade_effects(nets, data, owner, rng)
        effects.to_csv(f'{out_dir}/upgrade_effects.csv', index = False)
        rows = []
        for m in nets:
            e = effects[f'upgrade_effect_{m.name}']
            sup = effects[(effects.n_train_plain >= 1000) & (effects.n_train_upgraded >= 1000)]
            rows.append({'model': m.name, 'cards': len(effects), 'mean_effect': e.mean(), 'share_negative': (sup[f'upgrade_effect_{m.name}'] < 0).mean(),
                         'share_below_m0.05': (sup[f'upgrade_effect_{m.name}'] < -0.05).mean()})
        summary['upgrade_effects'] = pd.DataFrame(rows)
        rare = run_rare_cells(nets, data)
        rare.to_csv(f'{out_dir}/rare_cells.csv', index = False)
        summary['rare_cells'] = summarise_rare(rare)

    if 'picks' in sections:
        print('Card picks')
        agree, cal = run_picks(nets)
        if agree is not None:
            agree.to_csv(f'{out_dir}/picks_summary.csv', index = False)
            cal.to_csv(f'{out_dir}/picks_calibration.csv', index = False)
            summary['picks'] = agree

    if 'agreement' in sections:
        print('Agreement between models on card and boss relic picks')
        agree, items = run_agreement(nets)
        agree.to_csv(f'{out_dir}/agreement_summary.csv', index = False)
        items.to_csv(f'{out_dir}/agreement_items.csv', index = False)
        summary['agreement'] = agree

    for k, v in summary.items():
        v.to_csv(f'{out_dir}/summary_{k}.csv', index = False)
        print(f'\n=== {k} ===')
        print(v.to_string(index = False))
    print(f'\nWritten to {out_dir}')
