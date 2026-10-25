'''
    Takes slay the spire 2020 data dump and processes it for modeling

    TODO:
      Fix incorrect cols
      group pretrain beta with finetune by adding ascension column
      handle_searing_blow_correctly
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
import json
import re
import multiprocessing
import itertools
import random

import pandas as pd
import numpy as np

from collections import Counter
from pathlib import Path
from sklearn.model_selection import train_test_split
from time import time
from zipfile import ZipFile

from name_utils import RENAMES
from run_replay import replay_run

# Heart kill is recorded at floor 56 (see winning_run_example.json), act 3 boss kill is ~51. Verify against the dump
HEART_MIN_FLOOR = 55

# Each run contributes this many rows (one per randomly sampled floor) to a destination. Rows per run are capped
# because training data is rows x ~921 bytes: finetuning has to fit in RAM to be shuffled, alpha is ~75M runs
FLOORS_PER_RUN = {'finetuning': 5, 'pre_training_beta': 2, 'pre_training_alpha': 1}

# Runs whose replayed final deck differs from the logged final deck by more than this many cards are dropped
# replay_stats.csv has the residual distribution to help tune this
MAX_REPLAY_RESIDUAL = 0

# Columns written to the training csvs: cards + relics (915), characters (4), floor, victory
N_TRAIN_COLS = 921

def combine_and_clean(d, s, combine = True):
  # d = destination
  # s = split
  # Lazy programming... putting this here instead of figuring out how to actually pass properly  
  data_dir = os.getcwd() + '/data'
  temp_dir = f'{data_dir}/training_data/{d}/{s}'
  file_list = [f'{temp_dir}/{file_name}' for file_name in os.listdir(temp_dir)]

  for file_name in file_list:

    if combine is True:
      temp_file = pd.read_csv(file_name, header = 0, dtype = 'int8')

      # Checks to ensure the file has the correct number of columns
      # Update to correct number of columns in future
      if temp_file.shape[1] == N_TRAIN_COLS:

        output_file = f'{temp_dir}/{s}.csv'
        # output_file = f'{temp_dir}/{s}.parquet'

        if Path(output_file).exists():      
          temp_file.to_csv(output_file, index = False, mode = 'a', header = False)

        else:
          temp_file.to_csv(output_file, index = False, mode = 'w')

      else:
        temp_file.to_csv(f'{os.getcwd()}/data/processing_failures/incorrect_columns_{d}_{int(time())}.csv', index = False)

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

    # victory is also true for beating the act 3 boss without keys, so require reaching act 4 (Heart kill)
    victory = current_run['event']['victory'] and current_run['event']['floor_reached'] >= HEART_MIN_FLOOR

    stats[f'{destination}_runs_kept'] += 1
    stats[f'{destination}_rows'] += len(rows)

    for floor, deck, relics in rows:
      modeling_data = dict(deck)
      modeling_data.update(Counter(relics))
      modeling_data['floor'] = floor
      modeling_data['character'] = current_run['event']['character_chosen'].lower()
      modeling_data['victory'] = victory
      modeling_data['destination'] = destination
      modeling_data['id'] = current_run['event']['play_id']

      model_data_storage.append(modeling_data)

  # Process lists into dictionaries with value 1 to ensure all items are captured
  card_dict = {re.sub('\n', '', card):1 for card in master_card_list}
  relic_dict = {re.sub('\n', '', relic):1 for relic in master_relic_list}

  # Merge both dictionaries together
  dummy_row = {**card_dict, **relic_dict}

  # Add on remaining needed columns
  dummy_row['id'] = 'dummy_row'
  dummy_row['character'] = 'unknown' # Setting to unknown to ensure every file has it
  dummy_row['victory'] = True
  dummy_row['floor'] = 0

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

  # Moving victory to the end -- standard practice for modeling
  model_data_storage = model_data_storage[[c for c in model_data_storage if c not in ['victory']] + ['victory']]

  # Changing card titles to reflect current naming
  model_data_storage = model_data_storage.rename(columns = RENAMES)

  # Files that happen to have no runs for a character would otherwise be missing that one hot column
  character_cols = ['defect', 'ironclad', 'the_silent', 'watcher']
  for character_col in character_cols:
    if character_col not in model_data_storage.columns:
      model_data_storage[character_col] = 0

  # Explicit column order so every file matches starting_decks.csv and the model input
  item_cols = [RENAMES.get(c, c) for c in list(card_dict) + list(relic_dict)]
  model_data_storage = model_data_storage[item_cols + ['floor'] + character_cols + ['id', 'destination', 'victory']]

  # Double checks we have the correct number of columns -- this error tends to occur with pre train alpha
  # As of now I can't figure out why this occurs, but only seems to happen with a large minority of files
  # Root cause is NOT incorrect character names or victory column issues
  # Until then issue is handled by writing out file as a processing error and NOT write out
  
  # destination and id are still attached at this point
  if model_data_storage.shape[1] != N_TRAIN_COLS + 2:
    failure_path = f'{os.getcwd()}/data/processing_failures/incorrect_col_count_failure_{int(time())}.csv'
    model_data_storage.to_csv(failure_path, index = False, )
  
  
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

    if len(run_ids) > 1:
      train_ids, test_ids = train_test_split(run_ids, test_size=0.2)

      train = temp_data[temp_data['id'].isin(train_ids)].drop('id', axis = 1).astype('int8')
      test = temp_data[temp_data['id'].isin(test_ids)].drop('id', axis = 1).astype('int8')

      train.to_csv(f'{data_dir}/training_data/{u_d}/train/{time_stamp}.csv', index = False)
      test.to_csv(f'{data_dir}/training_data/{u_d}/test/{time_stamp}.csv', index = False)
    # If there's only one run in a file write it to train
    else:
      temp_data.drop('id', axis = 1).astype('int8').to_csv(f'{data_dir}/training_data/{u_d}/train/{time_stamp}.csv', index = False)

  return stats

if __name__ == '__main__':
  # NOTE: N_TRAIN_COLS = number of cols when accounting for cards, relics, character, floor, and victory cols

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

  destinations = ['finetuning', 'pre_training_beta', 'pre_training_alpha']
  splits = ['train', 'test']

  # Removes any files already in directories
  print('Cleaning Up Old Files')
  pool_obj = multiprocessing.Pool(num_cores)
  items = list(itertools.product(destinations, splits, [False]))
  output = pool_obj.starmap(combine_and_clean, items)
  pool_obj.close()

  # Pull list of all files to iterate through
  file_list = [data_dir + '/raw/' + f for f in os.listdir(data_dir + '/raw/')]

  k = 0
  total_stats = Counter()

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
  
  # Appends all of the individual files together into one, single file then cleans up the temp files
  # Runs serially to prevent data races against the main aggregation files
  print('aggregating files')
  
  destinations = ['finetuning', 'pre_training_beta', 'pre_training_alpha']
  splits = ['train', 'test']

  items = list(itertools.product(destinations, splits))

  pool_obj = multiprocessing.Pool(num_cores)
  run_progress = pool_obj.starmap(combine_and_clean, items)
  pool_obj.close()
  

  pd.Series(total_stats).sort_index().to_csv(f'{data_dir}/processing_failures/replay_stats.csv', header = ['count'])
