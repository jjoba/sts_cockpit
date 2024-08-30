import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset

import pdb

class CSVDataset(Dataset):
    def __init__(self, file_path, chunk_size = 1000):

        # Storing info passed
        self.file_path = file_path # path to file
        self.chunk_size = chunk_size # number of rows to read into memory at once from file

        # Setting up the reader as an iterator to enable get_chunk functionality
        self.reader = pd.read_csv(self.file_path, dtype = 'int8', header = 0, iterator = True)

        # Computing here on init for efficiency since __len__ gets called multiple times
        series = pd.read_csv(self.file_path, usecols = [0], dtype = 'int8', header = 0)
        self.LENGTH = series.shape[0]

        # Trackers for data file refreshes
        self.total_row_requests = 0 # total number of rows that have been requests. Governs reseting the reader
        self.chunk_row_requests = 0 # total number of rows requested from a chunk. Governs fetching new rows

        # Pre load the first chunk
        self.chunk = self.reader.get_chunk(chunk_size)
        self.chunk_rows = self.chunk.shape[0] # calculting dynamically in case the rows available are less than chunk size

    def __len__(self):

        return self.LENGTH
    
    def __getitem__(self, idx):
        # Send the X and y values for one row
        # NOTE: idx a.k.a. shuffling from data loader does nothing right now

        # Send single row
        if self.chunk_row_requests < self.chunk_rows:
            
            row = self.chunk.iloc[[self.chunk_row_requests], :]
            self.chunk_row_requests += 1
            self.total_row_requests += 1
        
        else:
            
            # Load new chunk only
            if self.total_row_requests < self.LENGTH:                
                
                # Load a new chunk
                self.chunk = self.reader.get_chunk(self.chunk_size)
                self.chunk_rows = self.chunk.shape[0]
                self.chunk_row_requests = 0

                # Send row and update counter
                row = self.chunk.iloc[[self.chunk_row_requests], :]
                self.chunk_row_requests += 1
                self.total_row_requests += 1

            # Resets reader for new epoch
            else:
                # Reset reader to start again at top of file
                self.reader = pd.read_csv(self.file_path, dtype = 'int8', header = 0, iterator = True)                
                self.total_row_requests = 0

                # Load a new chunk
                self.chunk = self.reader.get_chunk(self.chunk_size)
                self.chunk_rows = self.chunk.shape[0]
                self.chunk_row_requests = 0

                # Send row and update counter
                row = self.chunk.iloc[[self.chunk_row_requests], :]
                self.chunk_row_requests += 1
                self.total_row_requests += 1

        # Extract necessary rows, convert to tensor, and ship to data loader
        X = torch.from_numpy(row.iloc[:, :-1].to_numpy().astype(np.int8))
        y = torch.from_numpy(row.iloc[:, -1:].to_numpy().astype(np.int8))

        return X, y
    
df = CSVDataset('C:/Users/jjoba/OneDrive - KNEX/Desktop/test_file.csv', 70)
for i in range(50):
    print(df.__getitem__(5))

