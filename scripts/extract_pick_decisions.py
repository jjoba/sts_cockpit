'''
Card pick decisions of A20 held-out (test split) runs for the pick part of scripts/interaction_bench.py. Run from the project root:

    python scripts/extract_pick_decisions.py

Writes data/bench/decisions_v24_X.npy (int8, one row per card reward: the 929 feature columns of the deck and relics on arriving at the reward's floor, then
the run state columns of run_state.py) and decisions_v24_M.npy (int16: picked card column or -1 for skip, up to 5 offered card columns padded with -1, Heart kill 0 / 1,
floor). Runs are selected like the finetuning destination of process_data.py (same filters, A20 only, test split by the hash of play_id, modded runs left out).
Rewards on floor 0 (Neow) are left out. Cards of the offer that are not in the master card list are left out of the offer, a pick that is not in it counts as a skip.
'''

import gzip
import hashlib
import json
import multiprocessing
import os
import re
import sys
from itertools import repeat
from zipfile import ZipFile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from name_utils import clean_cards
from process_data import MAX_REPLAY_RESIDUAL, is_modded, is_test_run, run_labels
from run_replay import replay_run
from run_state import EXTRA_COLS, run_state
import random

ROOT = os.getcwd()
N_FEATURES = 929
N_OFFER = 5

# Keep 1 in RUN_ONE_IN of the test runs (picked from the hash bits process_data's split does not use) so the files stay a few GB
RUN_ONE_IN = int(os.environ.get('RUN_ONE_IN', 4))

def finetuning_run(e):
    # Same routing as process_data.process_runs for the finetuning destination
    return (not e.get('is_trial') and not e.get('is_daily') and not e.get('chose_seed') and e.get('timestamp', 0) >= 1593576000
            and e.get('is_beta') is False and 'character_chosen' in e and not e.get('is_endless') and e.get('ascension_level', 0) >= 20 and not is_modded(e))

def process_file(args):
    blob, cards, relics = args
    index = {c: i for i, c in enumerate(cards + relics)}
    card_set = set(cards)
    relic_set = set(relics)
    X, M = [], []
    n_runs = 0
    for run in json.loads(gzip.decompress(blob)):
        e = run['event']
        try:
            if not finetuning_run(e) or not is_test_run(e['play_id']) or (int(hashlib.md5(str(e['play_id']).encode()).hexdigest(), 16) // 5) % RUN_ONE_IN != 0:
                continue
        except (KeyError, TypeError):
            continue
        n_runs += 1
        floor_reached = e.get('floor_reached')
        if not isinstance(floor_reached, int):
            continue
        replay = replay_run(e, card_set, relic_set, floor_reached, random.Random(e['play_id']))
        if replay is None or replay[1] > MAX_REPLAY_RESIDUAL:
            continue
        by_floor = {floor: (deck, rel) for floor, deck, rel in replay[0]}
        for c in e.get('card_choices', []):
            f = c.get('floor')
            if not isinstance(f, int) or f < 1 or f not in by_floor:
                continue
            deck, rel = by_floor[f]
            offered = [index.get(n) for n in clean_cards([c.get('picked', 'SKIP')] + list(c.get('not_picked') or []))]
            offered = [i for i in offered if i is not None and i < len(cards)]
            picked_names = clean_cards([c.get('picked', 'SKIP')])
            picked = index.get(picked_names[0], -1) if picked_names else -1
            if picked >= len(cards):
                picked = -1
            if len(offered) == 0 or len(offered) > N_OFFER:
                continue
            row = np.zeros(N_FEATURES + len(EXTRA_COLS), dtype = np.int8)
            for k, v in deck.items():
                if k in index:
                    row[index[k]] = v
            for r in rel:
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
    return X, M, n_runs

if __name__ == '__main__':
    cards = [l.strip() for l in open(f'{ROOT}/data/master_lists/master_card_list.txt')]
    relics = [l.strip() for l in open(f'{ROOT}/data/master_lists/master_relic_list.txt')]
    assert len(cards) + len(relics) + 5 == N_FEATURES, len(cards) + len(relics)
    files = sorted(f'{ROOT}/data/raw/{f}' for f in os.listdir(f'{ROOT}/data/raw') if f.endswith('.zip'))
    if len(sys.argv) > 1:
        files = files[:int(sys.argv[1])]
    Xs, Ms, total_runs = [], [], 0
    pool = multiprocessing.Pool(max(1, multiprocessing.cpu_count() - 1))
    for k, zf in enumerate(files):
        z = ZipFile(zf)
        blobs = [z.read(n) for n in z.namelist() if 'gz' in n]
        for X, M, n in pool.imap_unordered(process_file, zip(blobs, repeat(cards), repeat(relics))):
            if X:
                Xs.append(np.stack(X)); Ms.append(np.array(M, dtype = np.int16))
            total_runs += n
        print(f'{k + 1}/{len(files)} zips, {total_runs} runs, {sum(len(m) for m in Ms)} decisions', flush = True)
    pool.close()
    X, M = np.concatenate(Xs), np.concatenate(Ms)
    os.makedirs(f'{ROOT}/data/bench', exist_ok = True)
    np.save(f'{ROOT}/data/bench/decisions_v24_X.npy', X)
    np.save(f'{ROOT}/data/bench/decisions_v24_M.npy', M)
    print('saved', X.shape, M.shape)
