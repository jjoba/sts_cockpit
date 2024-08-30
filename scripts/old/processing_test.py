'''
    Takes slay the spire 2020 data dump and processes it for modeling
'''

import os
import zipfile
import gzip
import json
import re

from collections import Counter

# Directory where all data is stored in raw, compressed format
data_dir = 'C:/Users/jjoba/sts/data/raw'

file_list = [data_dir + '/' + f for f in os.listdir(data_dir)]

print(file_list[0])

# Extract ip archive and dump to temp folder - note: extraction takes some time, would be wise to skip if possible
with zipfile.ZipFile(file_list[0], 'r') as zip_ref:
    zip_ref.extractall('C:/Users/jjoba/sts/data/raw_temp')

'''
    Come back to this step later

    [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser("~/files")) for f in fn]

'''

# gets all of the paths in the extracted folder
print([os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser('C:/Users/jjoba/sts/data/raw_temp')) for f in fn])


# Change location to temp folder and pull all files
data_dir = 'C:/Users/jjoba/sts/data/raw_temp'

file_list = [data_dir + '/' + f for f in os.listdir(data_dir)]

print(file_list[0])

# gz jason file
current_file = 'C:/Users/jjoba/sts/data/raw_temp/STS Data/2020-09-23-05-04#1019.json.gz'

# Reads as bytes file type
with gzip.open(current_file, 'rb') as f:
  file_content = f.read()
 
print(file_content[0])
print(type(file_content))

# Convert to json for easier manipulation
as_json = json.loads(file_content.decode('utf-8'))
print(as_json[0])
print(len(as_json)) # expected length = 1019 (see #XXX(X) in file name)

# Extract the first run
current_run = as_json[0]
print(current_run['event'].keys())

# write out the current run to inspect in editor
with open('C:/Users/jjoba/sts/data/test_run.json', 'w') as f:
    json.dump(current_run['event'], f)

# Conform card names to remove '+1' from upgrade, also treats all searing blows the same
# Force to lowercase
master_deck = current_run['event']['master_deck']
master_deck = [re.sub(r'\+.', '_u', card_name).lower() for card_name in master_deck]

# QTY of each card in the final deck
card_counts = Counter(master_deck)

# Append on character name and victory to create modeling data
modeling_data = card_counts
modeling_data['character'] = current_run['event']['character_chosen'].lower()
modeling_data['victory'] = current_run['event']['victory']

# Logic field determining which dataset this should go to
destination = 'finetuning'
if current_run['event']['is_trial'] is True:
   destination = 'pre-training alpha'

if current_run['event']['is_daily'] is True:
   destination = 'pre-training alpha'

if current_run['event']['chose_seed'] is True:
   destination = 'pre-training alpha'

# Pre watcher update data
if current_run['event']['timestamp'] < 1593576000:
   destination = 'pre-training alpha'

if current_run['event']['is_beta'] is True:
   destination = 'pre-training alpha'

if current_run['event']['is_endless'] is True:
   destination = 'pre-training alpha'

if destination == 'finetuning':
    if current_run['event']['ascension_level'] < 20:
        destination = 'pre-training beta'

print(destination)