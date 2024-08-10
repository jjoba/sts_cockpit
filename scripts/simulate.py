'''
    Trains model to predict how likely a given deck is to win
    TODO:
        Add non-finetuning data
        increase model capacity
'''

import pandas as pd
import numpy as np
import torch
from torch import nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Directory where all data is stored in raw, compressed format
project_dir = 'C:/Users/jjoba/sts/'

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # More layers seems to work worse
            nn.Linear(700, 700),
            nn.ReLU(),
            nn.Linear(700, 700),
            nn.ReLU(),
            nn.Linear(700, 700),
            nn.ReLU(),
            nn.Linear(700, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)

loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X.float())
        loss = loss_fn(pred, y.unsqueeze(1))

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X.float())
            test_loss += loss_fn(pred, y.unsqueeze(1)).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Load model
model_path = f'{project_dir}/model_objects/dev/fnn_v2.pth'
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load(model_path))

# read in the data
scenarios = pd.read_csv(f'{project_dir}/data/simulations/ironclad_run.csv')
scenarios.fillna(0, inplace = True)
print(scenarios)

# Check to see if the model actually learned anything
# Row row is silent based deck
# Row 2 is a lean discard/poison deck
# Row 2 should be >> Row 1

# For starting_decks.csv
# 0 = silent, 1 = ironcled, 3 = defect, 4 = watcher
with torch.no_grad():
    x = torch.from_numpy(scenarios.to_numpy().astype(np.float32)).to(device)
    pred = model(x)
    print(f'Predicted: "{pred}"')
