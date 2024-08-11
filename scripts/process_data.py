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
import zipfile
import gzip
import json
import re
import multiprocessing
import itertools

import pandas as pd

from collections import Counter
from pathlib import Path
from sklearn.model_selection import train_test_split

import pdb

def extract_to_raw_temp(f, p):
  # extracts zip file to raw temp
  # f = path to file for extraction
  # p = path to have raw_temp added
  with zipfile.ZipFile(f, 'r') as zip_ref:
    zip_ref.extractall(p + '/raw_temp')

def combine_and_clean(d, s):
  # d = destination
  # s = split
  # Lazy programming... putting this here instead of figuring out how to actually pass properly  
  data_dir = os.getcwd() + '/data'
  temp_dir = f'{data_dir}/training_data/{d}/{s}'
  file_list = [f'{temp_dir}/{file_name}' for file_name in os.listdir(temp_dir)]

  for file_name in file_list:
    temp_file = pd.read_parquet(file_name, engine = 'fastparquet')

    output_file = f'{temp_dir}/{s}.parquet'

    if Path(output_file).exists():      
      # temp_file.to_csv(output_file, index = False, mode = 'a', header = False)
      temp_file.to_parquet(output_file, index = False, compression = 'snappy', engine = 'fastparquet', append = True)

    else:
      # temp_file.to_csv(output_file, index = False, mode = 'w')
      temp_file.to_parquet(output_file, index = False, compression = 'snappy', engine = 'fastparquet')

    
    os.remove(file_name)

  return 0

def process_runs(jf):
  # Lazy programming... putting this here instead of figuring out how to actually pass properly  
  data_dir = os.getcwd() + '/data'

  # processes runs from some json file
  # f = path
  
  # Reads in all runs in the file as bytes
  with gzip.open(jf, 'rb') as f:
    runs_bytes = f.read()
  
  # Converts to json object to it's easier to work with
  runs_json = json.loads(runs_bytes.decode('utf-8'))

  # Holds all extracted runs from a given json file
  model_data_storage = []

  for current_run in runs_json:
    # Conform card names to remove '+1' from upgrade, also treats all searing blows the same
    # Force to lowercase
    master_deck = current_run['event']['master_deck']
    # Handles starter cards with _X at the end
    master_deck = [re.sub(r'_.{1}', '', card_name).lower() for card_name in master_deck]    
    master_deck = [re.sub(r'\+.*$', '_u', card_name) for card_name in master_deck]

    # First regex didn't catch all searing blows. Forcing them to all become the same
    # Keys off the fact that it's the only card with a number in it
    # master_deck = ['searing_blow_u' if 'searing_blow_u' in card_name else card_name for card_name in master_deck]
    master_deck = [re.sub(' ', '_', card_name) for card_name in master_deck]

    # QTY of each card in the final deck
    card_counts = dict(Counter(master_deck))

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
      with open(f'{os.getcwd()}/data/processing_failures/character_chosen_failure.json', 'w') as fout:
        json.dump(current_run, fout)
      current_run['event']['character_chosen'] = 'unknown'
      destination = 'pre_training_alpha'

    if current_run['event']['is_endless'] is True:
      destination = 'pre_training_alpha'

    if destination == 'finetuning':
        if current_run['event']['ascension_level'] < 20:
            destination = 'pre_training_beta'

    # Append on character name and victory to create modeling data
    modeling_data = card_counts
    modeling_data['character'] = current_run['event']['character_chosen'].lower()
    modeling_data['victory'] = current_run['event']['victory']
    modeling_data['destination'] = destination
    
    # add onto the model data storage
    id = current_run['event']['play_id']
    #model_data_storage[id] = modeling_data
    modeling_data['id'] = id

    model_data_storage.append(modeling_data)

  # Pull in the master card list to ensure each file has the same schema
  f = open(f'{data_dir}/master_card_list.txt', 'r')
  card_list = f.readlines()
  f.close()
  
  dummy_row = {re.sub('\n', '', card):1 for card in card_list}
  dummy_row['id'] = 'dummy_row'
  dummy_row['character'] = 'unknown' # Setting to unknown to ensure every file has it
  dummy_row['victory'] = True

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
  
  # Remove dummy row and drop id column
  model_data_storage = model_data_storage[model_data_storage['id'] != 'dummy_row']
  model_data_storage = model_data_storage.drop('id', axis = 1)

  # Moving victory to the end -- standard practice for modeling
  model_data_storage = model_data_storage[[c for c in model_data_storage if c not in ['victory']] + ['victory']]

  # Changing card titles to reflect current naming
  model_data_storage = model_data_storage.rename(columns={
    'underhanded_strike':'sneaky_strike', 
    'underhanded_strike_u':'sneaky_strike_u',
    'wireheading':'foresight', 
    'wireheading_u':'foresight_u', 
    'conserve_battery':'charge_battery',
    'conserve_battery':'charge_battery_u', 
    'lockon':'bullseye',
    'lockon_u':'bullseye_u',
    'undo':'equilibrium',
    'undo_u':'equilibrium_u',
    'steam_power': 'steam_barrier',
    'steam_power_u':'steam_barrier_u'
  })

  # Write out files
  # goes destination, then train/test split for pytorch
  
  # Ensures a given destination is in a file to avoid zero row situations
  unique_destinations = model_data_storage['destination'].unique()

  # Loop through each of unique destinations (finetuning, etc.)
  for u_d in unique_destinations:

    # Remove destination column and reduce data footprint
    temp_data = model_data_storage[model_data_storage['destination'] == u_d]
    temp_data = temp_data.drop('destination', axis = 1)
    temp_data = temp_data.astype('int8')

    if temp_data.shape[0] > 1:
      train, test = train_test_split(temp_data, test_size=0.2)

      train.to_parquet(f'{data_dir}/training_data/{u_d}/train/{re.sub('\\..+|.+STS Data.{1}|Monthly.{9}', '', jf)}.parquet', engine = 'fastparquet', index = False, compression = 'snappy')
      test.to_parquet(f'{data_dir}/training_data/{u_d}/train/{re.sub('\\..+|.+STS Data.{1}|Monthly.{9}', '', jf)}.parquet', engine = 'fastparquet', index = False, compression = 'snappy')

      # train.to_csv(f'{data_dir}/training_data/{u_d}/train/{re.sub('\\..+|.+STS Data.{1}|Monthly.{9}', '', jf)}.csv', index = False)
      # test.to_csv(f'{data_dir}/training_data/{u_d}/test/{re.sub('\\..+|.+STS Data.{1}|Monthly.{9}', '', jf)}.csv', index = False)
    # If there's only one row in a file write it to train
    else:
      train.to_parquet(f'{data_dir}/training_data/{u_d}/train/{re.sub('\\..+|.+STS Data.{1}|Monthly.{9}', '', jf)}.parquet', engine = 'fastparquet', index = False, compression = 'snappy')
      # temp_data.to_csv(f'{data_dir}/training_data/{u_d}/train/{re.sub('\\..+|.+STS Data.{1}|Monthly.{9}', '', jf)}.csv', index = False)

  # Ends function, nothing needed from this value
  return jf

if __name__ == '__main__':
  
  num_cores = multiprocessing.cpu_count() - 1
  
  # Directory where all data is stored in raw, compressed format
  data_dir = os.getcwd() + '/data'

  output = []

  # Hack to get the list of all unique card names
  card_list = []

  # Pull list of all files to iterate through
  file_list = [data_dir + '/raw/' + f for f in os.listdir(data_dir + '/raw/')]

  k = 0

  # Extract ip archive and dump to temp folder - note: extraction takes some time, would be wise to skip if possible
  for file in file_list:
    print(str(k / len(file_list)) + ' percent complete')
    # Unzips file
    extract_to_raw_temp(file, data_dir)

    # Recursively pulls all compressed json files
    json_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(data_dir + '/raw_temp')) for f in fn if 'json.gz' in f]

    pool_obj = multiprocessing.Pool(num_cores)
    run_progress = pool_obj.map(process_runs, json_files)
    pool_obj.close()

    
    '''
    for i in temp_output:
      card_list += [k for k in i.keys()]
    
    card_list = [k for k in set(card_list)]
    '''    
    
    # Clean up the files created
    for jf in json_files:
      os.remove(jf)
    
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
