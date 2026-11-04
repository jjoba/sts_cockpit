'''
    Takes slay the spire 2020 data dump and processes it for modeling

    TODO:
      Fix incorrect cols
      group pretrain beta with finetune by adding ascension column
      add relics
      add neow

    underhanded_strike is sneaky_strike
    truecontrol??
    stepandstrike??
    wireheading is foresight
    venomology was removed

    cards I can't find: pressure points, tranquility, recursion, claw, overclock

    Which card should I add? ironclad base deck
    Predicted Win Rate: "tensor([[3.0150e-07],
        [2.1605e-07],
        [2.2700e-05]])"
    ['current_deck', 'searing_blow', 'searing_blow_u']
'''

import os
import gzip
import hashlib
import json
import re
import multiprocessing
import itertools
import random
import shutil
import sys

import pandas as pd
import numpy as np

from collections import Counter
from pathlib import Path
from time import time
from zipfile import ZipFile

from name_utils import RENAMES
from run_replay import replay_run
from run_state import EXTRA_COLS, run_state

# Heart kill is recorded at floor 56 (see winning_run_example.json), act 3 boss kill is ~51. Verify against the dump
HEART_MIN_FLOOR = 55

# Each run contributes this many rows (one per randomly sampled floor) to a destination. Rows per run are capped
# because training data is rows x N_TRAIN_COLS bytes (raw int8, no header): finetuning has to fit in RAM to be shuffled, alpha is ~75M runs
# Destinations to write, comma separated in the DESTINATIONS env var (default all), e.g. DESTINATIONS=finetuning skips the alpha / beta runs entirely
DESTINATIONS = os.environ.get('DESTINATIONS', 'finetuning,pre_training_beta,pre_training_alpha').split(',')

FLOORS_PER_RUN = {'finetuning': 5, 'pre_training_beta': 1, 'pre_training_alpha': 1}

# Runs whose replayed final deck differs from the logged final deck by more than this many cards are dropped
# replay_stats.csv has the residual distribution to help tune this
MAX_REPLAY_RESIDUAL = 0

# 1 run in TEST_ONE_IN goes to the test split. Membership comes from a hash of the run's play_id (not Python's salted hash()),
# so it is the same on every processing run and can be recomputed later for any raw run, e.g. to evaluate on held out runs only
TEST_ONE_IN = 5

def is_test_run(play_id):
  return int(hashlib.md5(str(play_id).encode()).hexdigest(), 16) % TEST_ONE_IN == 0

# Prediction targets (heads of the network), in the order they are written as the LAST columns of every row. Heart kill must stay last
# 1 = the run went on to hit the milestone, 0 = it did not, -1 = the row has no answer to learn from (ignored in the loss), either because
# it is already past the milestone or the milestone does not exist for the run
# Each milestone means the same event for every run, whatever its ascension, so rows from different ascensions can be pooled and trained on together
LABEL_COLS = ['beat_act1_boss', 'beat_act2_boss', 'beat_act3_boss', 'beat_act3_second_boss', 'heart_kill']
N_FEATURE_COLS = 929

# Columns written to the training files: cards + relics (924), floor, characters (4), then the run state columns (HP, max HP, gold, see run_state.py), then the labels
N_TRAIN_COLS = N_FEATURE_COLS + len(EXTRA_COLS) + len(LABEL_COLS)

# Folder under data/ the training files go to. The v24 layout has the extra columns, so it never writes over data/training_data (the 929 column layout)
TRAINING_DIR = os.environ.get('TRAINING_DIR', 'training_data_v24')

# Floor a run is on once it has beaten each boss (bosses are fought on floors 16, 33 and 50, which is where deaths pile up in the data)
# beat_act3_boss is the FIRST act 3 boss, the only one below A20. A20 has a second act 3 boss (floor 51) which is its own milestone, beat_act3_second_boss,
# that only exists at A20 (-1 for every row of a run below A20). A run with no ascension_level counts as below A20
# A no key victory ends the run on floor 51 below A20 and floor 52 at A20
BOSS_CLEARED_FLOORS = [17, 34, 51]
ACT_3_SECOND_BOSS_CLEARED_FLOOR = 52

# Runs with values no normal run reaches (max hp or gold edited by mods / save editing) are not real gameplay and have far higher win rates.
# In a sample of A20 runs, max hp above 200 had a 54% Heart rate (3% overall) and gold above 3,000 25%. They go to alpha pretraining, not finetuning or beta
MODDED_MAX_HP = 250
MODDED_GOLD = 3000

def is_modded(event):
  # Checked on the whole run, not only the sampled floors, so a run is moved as a whole
  def peak(values):
    return max([v for v in values if isinstance(v, (int, float))], default = 0) if isinstance(values, list) else 0

  return peak(event.get('max_hp_per_floor')) > MODDED_MAX_HP or peak(event.get('gold_per_floor')) > MODDED_GOLD

def run_labels(event, floor):
  # Labels of one training row, in LABEL_COLS order, for a run's row sampled at this floor
  # Milestone labels are about the future: a row on or after the floor the milestone is cleared has the answer built in, so it is -1
  floor_reached = event['floor_reached']
  victory = bool(event['victory'])

  # victory is also true when ending the run by beating the act 3 boss(es) without keys, so it clears every boss milestone
  labels = [-1 if floor >= cleared_floor else int(victory or floor_reached >= cleared_floor) for cleared_floor in BOSS_CLEARED_FLOORS]

  if event.get('ascension_level', 0) >= 20 and floor < ACT_3_SECOND_BOSS_CLEARED_FLOOR:
    labels.append(int(victory or floor_reached >= ACT_3_SECOND_BOSS_CLEARED_FLOOR))
  else:
    labels.append(-1)

  # victory is also true for beating the act 3 boss without keys, so require reaching act 4 (Heart kill)
  labels.append(int(victory and floor_reached >= HEART_MIN_FLOOR))
  return labels

def combine_and_clean(d, s, combine = True):
  # d = destination
  # s = split
  # Lazy programming... putting this here instead of figuring out how to actually pass properly  
  data_dir = os.getcwd() + '/data'
  temp_dir = f'{data_dir}/{TRAINING_DIR}/{d}/{s}'
  file_list = [f'{temp_dir}/{file_name}' for file_name in os.listdir(temp_dir)]
  output_file = f'{temp_dir}/{s}.bin'

  for file_name in file_list:

    if combine is True:
      # Per file outputs are raw int8 rows N_TRAIN_COLS wide with no header, so a file that isn't a whole number of rows is bad
      if os.path.getsize(file_name) % N_TRAIN_COLS == 0:
        with open(file_name, 'rb') as fin, open(output_file, 'ab') as fout:
          shutil.copyfileobj(fin, fout, 64 * 1024 * 1024)

      else:
        shutil.move(file_name, f'{os.getcwd()}/data/processing_failures/incorrect_columns_{d}_{int(time())}.bin')
        continue

    os.remove(file_name)

  return 0

def process_runs(gzip_file, data_dir, master_card_list, master_relic_list):

  # gzip_file = gzipped binary string of a gzipped json file
  runs = json.loads(gzip.decompress(gzip_file))

  # Names allowed as columns, anything else in a run is ignored rather than breaking the schema for the whole file
  card_set = {re.sub('\n', '', card) for card in master_card_list}
  relic_set = {re.sub('\n', '', relic) for relic in master_relic_list}

  # Counts of what happened to runs in this file, summed across files in main
  stats = Counter()

  # Holds all extracted rows from a given json file
  model_data_storage = []

  for current_run in runs:
    # Logic field determining which dataset this should go to
    destination = 'finetuning'
    if current_run['event']['is_trial'] is True:
      destination = 'pre_training_alpha'

    if current_run['event']['is_daily'] is True:
      destination = 'pre_training_alpha'

    if current_run['event']['chose_seed'] is True:
      destination = 'pre_training_alpha'

    # Pre watcher update data
    if current_run['event']['timestamp'] < 1593576000:
      destination = 'pre_training_alpha'

    # Some files do not have the is beta flag - I don't trust it
    if 'is_beta' in current_run['event'].keys():
      if current_run['event']['is_beta'] is True:
        destination = 'pre_training_alpha'
    else:
      destination = 'pre_training_alpha'

    # Some runs do not have a character chosen
    # Seems to be from older data
    if 'character_chosen' not in current_run['event'].keys():
      with open(f'{os.getcwd()}/data/processing_failures/character_chosen_failure_{int(time())}.json', 'w') as fout:
        json.dump(current_run, fout)
      current_run['event']['character_chosen'] = 'unknown'
      destination = 'pre_training_alpha'

    if current_run['event']['is_endless'] is True:
      destination = 'pre_training_alpha'

    if destination == 'finetuning':
        if current_run['event']['ascension_level'] < 20:
            destination = 'pre_training_beta'

    if destination in ('finetuning', 'pre_training_beta') and is_modded(current_run['event']):
      destination = 'pre_training_alpha'
      stats['modded_moved_to_alpha'] += 1

    if destination not in DESTINATIONS:
      continue

    stats[f'{destination}_runs'] += 1

    # Rebuild deck & relics at sampled floors
    rng = random.Random(current_run['event']['play_id'])
    replay = replay_run(current_run['event'], card_set, relic_set, FLOORS_PER_RUN[destination], rng)

    if replay is None:
      stats[f'{destination}_dropped_not_replayable'] += 1
      continue

    rows, residual = replay

    # Residual distribution is across all replayable runs so MAX_REPLAY_RESIDUAL can be tuned
    stats['residual_' + ('0' if residual == 0 else '1' if residual == 1 else '2-3' if residual < 4 else '4-6' if residual < 7 else '7+')] += 1

    if residual > MAX_REPLAY_RESIDUAL:
      stats[f'{destination}_dropped_residual'] += 1
      continue

    stats[f'{destination}_runs_kept'] += 1
    stats[f'{destination}_rows'] += len(rows)

    for floor, deck, relics in rows:
      modeling_data = dict(deck)
      modeling_data.update(Counter(relics))
      modeling_data.update(run_state(current_run['event'], floor))
      modeling_data['floor'] = floor
      modeling_data['character'] = current_run['event']['character_chosen'].lower()
      modeling_data.update(dict(zip(LABEL_COLS, run_labels(current_run['event'], floor))))
      modeling_data['destination'] = destination
      modeling_data['id'] = current_run['event']['play_id']

      model_data_storage.append(modeling_data)

  # A file with no rows for the DESTINATIONS being written has nothing to write
  if len(model_data_storage) == 0:
    return stats

  # Process lists into dictionaries with value 1 to ensure all items are captured
  card_dict = {re.sub('\n', '', card):1 for card in master_card_list}
  relic_dict = {re.sub('\n', '', relic):1 for relic in master_relic_list}

  # Merge both dictionaries together
  dummy_row = {**card_dict, **relic_dict}

  # Add on remaining needed columns
  dummy_row['id'] = 'dummy_row'
  dummy_row['character'] = 'unknown' # Setting to unknown to ensure every file has it
  dummy_row.update({label: 1 for label in LABEL_COLS})
  dummy_row['floor'] = 0
  dummy_row.update({c: 0 for c in EXTRA_COLS})

  # model_data_storage_save = model_data_storage.copy()

  # Add on the dummy row - all as a single list
  model_data_storage = [dummy_row] + model_data_storage

  dummy_2 = dummy_row.copy()
  dummy_2['character'] = 'watcher'
  model_data_storage = [dummy_2] + model_data_storage

  # Convert to dataframe
  model_data_storage = pd.DataFrame(model_data_storage)

  # Force all NAs to 0
  model_data_storage.fillna(0, inplace = True)

  # one hot encode character
  one_hot = pd.get_dummies(model_data_storage['character'])
  model_data_storage = model_data_storage.drop('character', axis = 1)
  model_data_storage = model_data_storage.join(one_hot) 

  # Setting unknown as the reference
  model_data_storage = model_data_storage.drop('unknown', axis = 1)
  
  # Remove dummy rows, id is kept until after the train / test split so a run's rows never land on both sides
  model_data_storage = model_data_storage[model_data_storage['id'] != 'dummy_row']

  # Moving the labels to the end -- standard practice for modeling
  model_data_storage = model_data_storage[[c for c in model_data_storage if c not in LABEL_COLS] + LABEL_COLS]

  # Changing card titles to reflect current naming
  model_data_storage = model_data_storage.rename(columns = RENAMES)

  # Files that happen to have no runs for a character would otherwise be missing that one hot column
  character_cols = ['defect', 'ironclad', 'the_silent', 'watcher']
  for character_col in character_cols:
    if character_col not in model_data_storage.columns:
      model_data_storage[character_col] = 0

  # Explicit column order so every file matches starting_decks.csv and the model input
  item_cols = [RENAMES.get(c, c) for c in list(card_dict) + list(relic_dict)]
  model_data_storage = model_data_storage[item_cols + ['floor'] + character_cols + EXTRA_COLS + ['id', 'destination'] + LABEL_COLS]

  # Double checks we have the correct number of columns -- this error tends to occur with pre train alpha
  # As of now I can't figure out why this occurs, but only seems to happen with a large minority of files
  # Root cause is NOT incorrect character names or victory column issues
  # Until then issue is handled by writing out file as a processing error and NOT write out
  
  # destination and id are still attached at this point
  if model_data_storage.shape[1] != N_TRAIN_COLS + 2:
    failure_path = f'{os.getcwd()}/data/processing_failures/incorrect_col_count_failure_{int(time())}.csv'
    model_data_storage.to_csv(failure_path, index = False, )
    return stats
  
  
  # Write out files
  # goes destination, then train/test split for pytorch
  
  # Ensures a given destination is in a file to avoid zero row situations
  unique_destinations = model_data_storage['destination'].unique()

  # Loop through each of unique destinations (finetuning, etc.)
  for u_d in unique_destinations:

    # Remove destination column
    temp_data = model_data_storage[model_data_storage['destination'] == u_d]
    temp_data = temp_data.drop('destination', axis = 1)

    time_stamp = int(time() + np.random.randint(0, 10000000000, 1))

    # Split on runs (not rows) since a run can have several rows
    run_ids = temp_data['id'].unique()

    # Floor is capped so endless runs can't wrap around int8
    temp_data = temp_data.assign(floor = temp_data['floor'].clip(upper = 127))

    test_ids = [run_id for run_id in run_ids if is_test_run(run_id)]
    train_ids = [run_id for run_id in run_ids if not is_test_run(run_id)]

    train = temp_data[temp_data['id'].isin(train_ids)].drop('id', axis = 1).astype('int8')
    test = temp_data[temp_data['id'].isin(test_ids)].drop('id', axis = 1).astype('int8')

    train.to_numpy().tofile(f'{data_dir}/{TRAINING_DIR}/{u_d}/train/{time_stamp}.bin')
    test.to_numpy().tofile(f'{data_dir}/{TRAINING_DIR}/{u_d}/test/{time_stamp}.bin')

  return stats

if __name__ == '__main__':
  # NOTE: N_TRAIN_COLS = number of cols when accounting for cards, relics, character, floor, and the label cols

  num_cores = multiprocessing.cpu_count() - 1
  
  # Directory where all data is stored in raw, compressed format
  data_dir = os.getcwd() + '/data'

  # Pull in the master card list to ensure each file has the same schema
  f = open(f'{data_dir}/master_lists/master_card_list.txt', 'r')
  master_card_list = f.readlines()
  f.close()

  # Pull in the master card list to ensure each file has the same schema
  f = open(f'{data_dir}/master_lists/master_relic_list.txt', 'r')
  master_relic_list = f.readlines()
  f.close()

  destinations = DESTINATIONS
  splits = ['train', 'test']

  # Output folders and a layout file the training code reads to find the extra columns
  for d, sp in itertools.product(destinations, splits):
    os.makedirs(f'{data_dir}/{TRAINING_DIR}/{d}/{sp}', exist_ok = True)
  with open(f'{data_dir}/{TRAINING_DIR}/layout.json', 'w') as f:
    json.dump({'n_base': N_FEATURE_COLS, 'extra_cols': EXTRA_COLS, 'n_labels': len(LABEL_COLS)}, f)

  # Removes any files already in directories
  print('Cleaning Up Old Files')
  pool_obj = multiprocessing.Pool(num_cores)
  items = list(itertools.product(destinations, splits, [False]))
  output = pool_obj.starmap(combine_and_clean, items)
  pool_obj.close()

  # Pull list of all files to iterate through
  file_list = [data_dir + '/raw/' + f for f in os.listdir(data_dir + '/raw/')]

  # Optional first argument limits the number of zips processed, for trial runs
  if len(sys.argv) > 1:
    file_list = sorted(file_list)[:int(sys.argv[1])]

  k = 0
  total_stats = Counter()

  # Zip sizes are used to project total output size and time from the zips processed so far
  zip_bytes = [os.path.getsize(f) for f in file_list]
  start_time = time()

  # Extract ip archive and dump to temp folder - note: extraction takes some time, would be wise to skip if possible
  for file in file_list:
    print(f'{(100 * (k / len(file_list))):>0.1f}% Complete')
    
    # Extracts file to memory
    input_zip = ZipFile(file)
    res = {name: input_zip.read(name) for name in input_zip.namelist()}

    # Gets list of gzipped json files
    json_files = [jf for jf in res.keys()]

    # There are some files in the directory that are not gzipped jsons of runs. We don't need those
    for jf in json_files:
        if 'gz' not in jf:
            print(f'non gzip file found: {jf}')
            del res[jf]

    #for v in res.values():
    #  process_runs(v, data_dir, master_card_list, master_relic_list)
    
    pool_obj = multiprocessing.Pool(num_cores)
    items = list(itertools.product(res.values(), [data_dir], [master_card_list], [master_relic_list]))
    run_progress = pool_obj.starmap(process_runs, items)
    pool_obj.close()

    for file_stats in run_progress:
      total_stats.update(file_stats)

    print(dict(total_stats))

    k += 1

    # Projects totals from the share of zip bytes processed so far
    share_done = sum(zip_bytes[:k]) / sum(zip_bytes)
    output_gb = sum(os.path.getsize(os.path.join(root, name)) for root, _, names in os.walk(f'{data_dir}/{TRAINING_DIR}') for name in names) / 1e9
    elapsed = time() - start_time
    print(f'{k}/{len(file_list)} zips: {output_gb:.1f} GB written so far, projected total {output_gb / share_done:.0f} GB')
    print(f'{elapsed / 60:.0f} min elapsed, projected total {elapsed / share_done / 60:.0f} min')
  
  # Appends all of the individual files together into one, single file then cleans up the temp files
  # Runs serially to prevent data races against the main aggregation files
  print('aggregating files')
  
  destinations = DESTINATIONS
  splits = ['train', 'test']

  items = list(itertools.product(destinations, splits))

  pool_obj = multiprocessing.Pool(num_cores)
  run_progress = pool_obj.starmap(combine_and_clean, items)
  pool_obj.close()
  

  pd.Series(total_stats).sort_index().to_csv(f'{data_dir}/processing_failures/replay_stats.csv', header = ['count'])
