'''
Item 12, stage 1 (extract): rebuild the finetuning rows exactly as process_data.py does (same filters, same replay, same sampled floors,
same residual filter) but keep run metadata next to each row, so confounders can be tested without touching the training files.
Writes to OUT_DIR only (X_<k>.npy int8 rows in the training layout, M_<k>.npy float64 metadata), nothing under data/training_data.
Run from the project root: python scripts/diag_confounders_extract.py <out_dir> [n_zips]
'''
import gzip
import json
import multiprocessing
import os
import random
import re
import sys
from collections import Counter
from time import time
from zipfile import ZipFile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from process_data import FLOORS_PER_RUN, MAX_REPLAY_RESIDUAL, is_modded, is_test_run, run_labels
from run_replay import replay_run

ROOT = os.getcwd()
CARDS = [re.sub('\n', '', l) for l in open(f'{ROOT}/data/master_lists/master_card_list.txt')]
RELICS = [re.sub('\n', '', l) for l in open(f'{ROOT}/data/master_lists/master_relic_list.txt')]
CARD_IDX = {n: i for i, n in enumerate(CARDS)}
RELIC_IDX = {n: len(CARDS) + i for i, n in enumerate(RELICS)}
FLOOR_COL = len(CARDS) + len(RELICS)
CHARS = ['defect', 'ironclad', 'the_silent', 'watcher']
N_X = FLOOR_COL + 1 + len(CHARS) + 5
CARD_SET, RELIC_SET = set(CARDS), set(RELICS)

# Metadata columns of M
M_COLS = ['is_test', 'run_key', 'player_experience', 'timestamp', 'floor', 'hp', 'max_hp', 'gold', 'n_smith_before', 'n_rest_before',
          'n_smith_total', 'floor_reached', 'neow_bonus', 'neow_cost']
NEOW_BONUS, NEOW_COST = {}, {}

def at_floor(arr, floor, default):
    # State on arriving at `floor` = what the per-floor list holds for the floor before it
    if not isinstance(arr, list) or floor < 2 or len(arr) < floor - 1:
        return default
    return arr[floor - 2]

def code(table, value):
    return table.setdefault(str(value), len(table))

def is_finetuning(e):
    if e.get('is_trial') is True or e.get('is_daily') is True or e.get('chose_seed') is True or e.get('is_endless') is True:
        return False
    if e['timestamp'] < 1593576000 or e.get('is_beta') is not False or 'character_chosen' not in e:
        return False
    return e.get('ascension_level', 0) >= 20

def process_file(blob):
    X, M = [], []
    n_runs = n_kept = n_modded = 0
    for run in json.loads(gzip.decompress(blob)):
        e = run['event']
        try:
            if not is_finetuning(e):
                continue
            if is_modded(e):
                n_modded += 1
                continue
            n_runs += 1
            replay = replay_run(e, CARD_SET, RELIC_SET, FLOORS_PER_RUN['finetuning'], random.Random(e['play_id']))
            if replay is None or replay[1] > MAX_REPLAY_RESIDUAL:
                continue
            smith_floors = sorted(c['floor'] for c in e.get('campfire_choices', []) if c.get('key') == 'SMITH')
            rest_floors = sorted(c['floor'] for c in e.get('campfire_choices', []) if c.get('key') == 'REST')
            run_key = int(e['play_id'].replace('-', '')[:15], 16)
            character = e['character_chosen'].lower()
            for floor, deck, relics in replay[0]:
                x = np.zeros(N_X, dtype = np.int8)
                for k, v in deck.items():
                    x[CARD_IDX[k]] = v
                for r in relics:
                    x[RELIC_IDX[r]] += 1
                x[FLOOR_COL] = min(floor, 127)
                x[FLOOR_COL + 1 + CHARS.index(character)] = 1
                x[-5:] = run_labels(e, floor)
                hp = at_floor(e.get('current_hp_per_floor'), floor, np.nan)
                max_hp = at_floor(e.get('max_hp_per_floor'), floor, np.nan)
                gold = at_floor(e.get('gold_per_floor'), floor, np.nan)
                M.append([int(is_test_run(e['play_id'])), run_key, e.get('player_experience', np.nan), e['timestamp'], floor, hp, max_hp, gold,
                          sum(f < floor for f in smith_floors), sum(f < floor for f in rest_floors), len(smith_floors), e['floor_reached'],
                          code(NEOW_BONUS, e.get('neow_bonus')), code(NEOW_COST, e.get('neow_cost'))])
                X.append(x)
            n_kept += 1
        except (KeyError, TypeError, ValueError, IndexError, AttributeError):
            continue
    return (np.array(X, dtype = np.int8).reshape(-1, N_X), np.array(M, dtype = np.float64).reshape(-1, len(M_COLS)), (n_runs, n_modded), n_kept, dict(NEOW_BONUS), dict(NEOW_COST))

if __name__ == '__main__':
    out = sys.argv[1]
    os.makedirs(out, exist_ok = True)
    zips = sorted(f'{ROOT}/data/raw/{f}' for f in os.listdir(f'{ROOT}/data/raw') if f.endswith('.zip'))
    if len(sys.argv) > 2:
        zips = zips[:int(sys.argv[2])]

    start = time()
    neow_bonus, neow_cost = {}, {}
    total_runs = total_kept = total_modded = 0
    pool = multiprocessing.Pool(max(1, multiprocessing.cpu_count() - 2))
    for k, z in enumerate(zips):
        if os.path.exists(f'{out}/X_{k}.npy'):
            continue
        zf = ZipFile(z)
        blobs = [zf.read(n) for n in zf.namelist() if n.endswith('.gz')]
        results = pool.map(process_file, blobs, chunksize = 4)
        # Neow codes are per worker dicts, remap to one shared table
        Xs, Ms = [], []
        for X, M, nr, nk, nb, nc in results:
            if len(X) == 0:
                continue
            inv_b, inv_c = {v: k2 for k2, v in nb.items()}, {v: k2 for k2, v in nc.items()}
            M[:, M_COLS.index('neow_bonus')] = [neow_bonus.setdefault(inv_b[int(c)], len(neow_bonus)) for c in M[:, M_COLS.index('neow_bonus')]]
            M[:, M_COLS.index('neow_cost')] = [neow_cost.setdefault(inv_c[int(c)], len(neow_cost)) for c in M[:, M_COLS.index('neow_cost')]]
            Xs.append(X)
            Ms.append(M)
        total_runs += sum(r[2][0] for r in results)
        total_modded += sum(r[2][1] for r in results)
        total_kept += sum(r[3] for r in results)
        if not Xs:
            print(f'{k + 1}/{len(zips)} zips: no finetuning rows', flush = True)
            continue
        np.save(f'{out}/X_{k}.npy', np.concatenate(Xs))
        np.save(f'{out}/M_{k}.npy', np.concatenate(Ms))
        print(f'{k + 1}/{len(zips)} zips, {total_runs} finetuning runs, {total_kept} kept, {total_modded} modded, {(time() - start) / 60:.0f} min', flush = True)
    json.dump({'M_COLS': M_COLS, 'neow_bonus': neow_bonus, 'neow_cost': neow_cost}, open(f'{out}/meta.json', 'w'))
