'''
    Script to create master list of cards, relics, etc.

    output written to data/master_lists
'''

import os
import gzip
import json
import multiprocessing

from zipfile import ZipFile

from name_utils import clean_cards, clean_relics

import pdb

def get_list(gzip_file):
    # gzip_file = gzipped binary string of a gzipped json file
    runs = json.loads(gzip.decompress(gzip_file))

    card_list = []
    relic_list = []

    for run in runs:
        # returns list items from 
        temp_run = run['event']

        master_deck = temp_run['master_deck']

        # Cleans up card names (see name_utils.py)
        master_deck = clean_cards(master_deck)

        # update the card list
        card_list += [c for c in set(master_deck)]
        
        # Process relics
        relics = temp_run['relics']

        relic_list += [r for r in set(clean_relics(relics))]

    # Sort lists and get unique values before returning
    card_list = [c for c in set(card_list)]
    card_list.sort()

    relic_list = [r for r in set(relic_list)]
    relic_list.sort()

    return card_list, relic_list

if __name__ == '__main__':
    num_cores = multiprocessing.cpu_count() - 1

    k= 0
  
    # Directory where all data is stored in raw, compressed format
    data_dir = os.getcwd() + '/data'

    # Pull list of all files to iterate through
    file_list = [data_dir + '/raw/' + f for f in os.listdir(data_dir + '/raw/')]

    # Storage buffers
    master_card_list = []
    master_relic_list = []

    for file in file_list:
        print(f'{(100 * (k / len(file_list))):>0.1f}%')
        # Unzip file to memory
        input_zip = ZipFile(file)
        res = {name: input_zip.read(name) for name in input_zip.namelist()}

        # Gets list of gzipped json files
        json_files = [jf for jf in res.keys()]

        # There are some files in the directory that are not gzipped jsons of runs. We don't need those
        for jf in json_files:
            if 'gz' not in jf:
                print(f'non gzip file found: {jf}')
                del res[jf]

        # Get all cards, relics, etc. from each json run
        pool_obj = multiprocessing.Pool(num_cores)
        # Each index of output is a run, sub indexed in oder by values returned from get list
        output = pool_obj.map(get_list, res.values())
        pool_obj.close()

        # Unpack, set, and sort
        file_card_list = [c for c_list in output for c in c_list[0]]
        file_card_list = [c for c in set(file_card_list)]
        file_card_list.sort()

        file_relic_list = [r for r_list in output for r in r_list[1]]
        file_relic_list = [r for r in set(file_relic_list)]
        file_relic_list.sort()

        # Update master list before moving on to next file
        master_card_list += file_card_list
        master_relic_list += file_relic_list

        k += 1

    # Set and sort
    master_card_list = [c for c in set(master_card_list)]
    master_card_list.sort()

    master_relic_list = [r for r in set(master_relic_list)]
    master_relic_list.sort()

    # Write out files
    with open(os.getcwd() + '/data/master_lists/master_card_list.txt', 'w') as out_file:
        for line in master_card_list:
            out_file.write(f'{line}\n')

    with open(os.getcwd() + '/data/master_lists/master_relic_list.txt', 'w') as out_file:
        for line in master_relic_list:
            out_file.write(f'{line}\n')