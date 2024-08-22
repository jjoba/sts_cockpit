'''
    Trains model to predict how likely a given deck is to win
    TODO:
        
'''
import torch
import os
import shutil

import pandas as pd
import numpy as np

from torch import nn
from torch.utils.data import Dataset, DataLoader

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
    
# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # More layers seems to work worse
            nn.Linear(863, 863),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(863, 863),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(863, 863),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(863, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.flatten(0,1), y.flatten()
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X.float())
        loss = loss_fn(pred, y.unsqueeze(1).float())

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 250 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.flatten(0,1), y.flatten()
            X, y = X.to(device), y.to(device)
            pred = model(X.float())
            test_loss += loss_fn(pred, y.unsqueeze(1).float()).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss


############### RUN ##############

if __name__ == '__main__':
    
    model_version = 'fnn_v14'
    
    # Run time variables
    # increase chunksize to speed training at the expense of memory usage
    # if chunk size > number of rows in file, then pandas reads the entire file w/out erroring
    chunk_size = 1000000
    batch_size = 64

    # true batch size is chunk_size * batch_size

    project_dir = os.getcwd()

    # How many epochs for each data set and in which order
    training_dict = [
        # {'destination': 'pre_training_alpha', 'epochs': 15},
        # {'destination': 'pre_training_beta', 'epochs': 20},
        {'destination': 'finetuning', 'epochs': 20}
    ]

    # set fallback option to CPU as some functions aren't implemented for MPS yet
    # Set the below command using export in terminal before running script
    # os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    # print(os.environ['PYTORCH_ENABLE_MPS_FALLBACK'])

    # Where is training to take place?
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f'Using device: {device}')

    # Instantiate Model
    model = NeuralNetwork().to(device)

    # Setup optimization
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    for regimen in training_dict:
        print(f'Training on destination: {regimen['destination']}')
        # Setup reading data for training and testing
        training_data = CSVDataset(f'{project_dir}/data/training_data/{regimen['destination']}/train/train.csv', chunk_size)
        training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

        testing_data = CSVDataset(f'{project_dir}/data/training_data/{regimen['destination']}/test/test.csv', chunk_size)
        testing_loader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)

        loss_tracker = []
        for t in range(regimen['epochs']):
            print(f"Epoch {t+1}\n-------------------------------")
            train(training_loader, model, loss_fn, optimizer)
            epoch_test_loss = test(testing_loader, model, loss_fn)

            loss_tracker += [epoch_test_loss]

            # Save out the model
            model_path = f'{project_dir}/model_objects/dev/{model_version}/{regimen['destination']}_epoch_{t}.pth'
            model_scripted = torch.jit.script(model) # Export to TorchScript
            model_scripted.save(model_path) # Save
            print(f"Saved PyTorch Model State to {model_path}")

        index_min = np.argmin(loss_tracker)
        print(f'Best Model Was Epoch: {index_min + 1} with a loss of {loss_tracker[index_min]}')
        print(loss_tracker)

        # Loads the best model into memory for use in the next training step
        # NOTE: this causes issue using MPS backed and requires fallback to CPU which is slow
        ## Disabling using the best model and instead uses the last epoch from pre-training which is probably close enough 
        # model_path = f'{project_dir}/model_objects/dev/{model_version}/{regimen['destination']}_epoch_{index_min}.pth'        
        # model = torch.jit.load(model_path).to(device)

    # Move the top performing finetuned model to production
    shutil.copy(
        f'{project_dir}/model_objects/dev/{model_version}/finetuning_epoch_{index_min}.pth',
        f'{project_dir}/model_objects/production/{model_version}/finetuning_epoch_{index_min}.pth' )

        