'''
Boss relic pick decisions of A20 held-out (test split) runs for the agreement part of scripts/interaction_bench.py. Run from the project root:

    python scripts/extract_relic_decisions.py

Writes data/bench/decisions_v24_relic_X.npy and decisions_v24_relic_M.npy in the layout of extract_pick_decisions.py: the 929 feature columns of the deck and relics before the
boss relic was taken (the picked relic is taken out of the replayed relic list) then the run state columns, and per decision the picked relic column (-1 for skip), up to 5 offered
relic columns, Heart kill 0 / 1 and the boss floor (17, 34 or 51). Same run selection as extract_pick_decisions.py. Only boss relics have a logged offer; relics from shops, chests
and events do not, so they cannot be scored as picks. Set PROCESSES to change the number of worker processes (default 4).
'''

import gzip
import json
import multiprocessing
import os
import random
import sys
from itertools import repeat
from zipfile import ZipFile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_pick_decisions import N_FEATURES, N_OFFER, RUN_ONE_IN, finetuning_run
from name_utils import clean_relics
from process_data import MAX_REPLAY_RESIDUAL, is_test_run, run_labels
from run_replay import BOSS_RELIC_FLOORS, replay_run
from run_state import EXTRA_COLS, run_state
import hashlib

ROOT = os.getcwd()

def process_file(args):
    blob, cards, relics = args
    index = {c: i for i, c in enumerate(cards + relics)}
    card_set, relic_set = set(cards), set(relics)
    X, M = [], []
    for run in json.loads(gzip.decompress(blob)):
        e = run['event']
        try:
            if not finetuning_run(e) or not is_test_run(e['play_id']) or (int(hashlib.md5(str(e['play_id']).encode()).hexdigest(), 16) // 5) % RUN_ONE_IN != 0:
                continue
        except (KeyError, TypeError):
            continue
        floor_reached = e.get('floor_reached')
        if not isinstance(floor_reached, int) or not e.get('boss_relics'):
            continue
        replay = replay_run(e, card_set, relic_set, floor_reached, random.Random(e['play_id']))
        if replay is None or replay[1] > MAX_REPLAY_RESIDUAL:
            continue
        by_floor = {floor: (deck, rel) for floor, deck, rel in replay[0]}
        for i, b in enumerate(e['boss_relics']):
            f = BOSS_RELIC_FLOORS[min(i, 2)]
            if f not in by_floor or f > floor_reached:
                continue
            deck, rel = by_floor[f]
            picked_name = clean_relics([b['picked']])[0] if b.get('picked') else None
            offered = [index.get(n) for n in clean_relics(([b['picked']] if b.get('picked') else []) + list(b.get('not_picked') or []))]
            offered = [j for j in offered if j is not None and j >= len(cards)]
            picked = index.get(picked_name, -1) if picked_name else -1
            if picked < len(cards):
                picked = -1
            if len(offered) < 2 or len(offered) > N_OFFER:
                continue
            row = np.zeros(N_FEATURES + len(EXTRA_COLS), dtype = np.int8)
            for k, v in deck.items():
                if k in index:
                    row[index[k]] = v
            removed = False
            for r in rel:
                if r == picked_name and not removed:
                    removed = True # the replay already holds the relic that was picked here
                    continue
                if r in index:
                    row[index[r]] += 1
            row[N_FEATURES - 5] = min(f, 127)
            ch = str(e['character_chosen']).lower()
            if ch in ('defect', 'ironclad', 'the_silent', 'watcher'):
                row[N_FEATURES - 4 + ['defect', 'ironclad', 'the_silent', 'watcher'].index(ch)] = 1
            st = run_state(e, f)
            for j, col in enumerate(EXTRA_COLS):
                row[N_FEATURES + j] = st[col]
            X.append(row)
            M.append([picked] + (offered + [-1] * N_OFFER)[:N_OFFER] + [run_labels(e, f)[-1], f])
    return X, M

if __name__ == '__main__':
    cards = [l.strip() for l in open(f'{ROOT}/data/master_lists/master_card_list.txt')]
    relics = [l.strip() for l in open(f'{ROOT}/data/master_lists/master_relic_list.txt')]
    files = sorted(f'{ROOT}/data/raw/{f}' for f in os.listdir(f'{ROOT}/data/raw') if f.endswith('.zip'))
    if len(sys.argv) > 1:
        files = files[:int(sys.argv[1])]
    Xs, Ms = [], []
    pool = multiprocessing.Pool(int(os.environ.get('PROCESSES', 4)))
    for k, zf in enumerate(files):
        z = ZipFile(zf)
        blobs = [z.read(n) for n in z.namelist() if 'gz' in n]
        for X, M in pool.imap_unordered(process_file, zip(blobs, repeat(cards), repeat(relics))):
            if X:
                Xs.append(np.stack(X)); Ms.append(np.array(M, dtype = np.int16))
        print(f'{k + 1}/{len(files)} zips, {sum(len(m) for m in Ms)} decisions', flush = True)
    pool.close()
    X, M = np.concatenate(Xs), np.concatenate(Ms)
    np.save(f'{ROOT}/data/bench/decisions_v24_relic_X.npy', X)
    np.save(f'{ROOT}/data/bench/decisions_v24_relic_M.npy', M)
    print('saved', X.shape, M.shape)
