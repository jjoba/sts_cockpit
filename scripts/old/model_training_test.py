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

import pdb


# Directory where all data is stored in raw, compressed format
project_dir = 'C:/Users/jjoba/sts/'

# read in the data
print('loading data')
finetune = pd.read_csv(f'{project_dir}/data/training_data/finetuning.csv', header = 0, on_bad_lines = 'warn', dtype = 'object')

print(finetune.shape)

print('cleaning data')
# There are some rows that are just the column name. Likely due to append not having any rows
finetune = finetune[finetune['a_thousand_cuts'] != 'a_thousand_cuts']

# Coerce T/F strings to bools then floats before main processing portion below
finetune['ironclad'] = finetune['ironclad'].replace({'True': 1.0, 'False': 0.0}).astype('float32')
finetune['the_silent'] = finetune['the_silent'].replace({'True': 1.0, 'False': 0.0}).astype('float32')
finetune['defect'] = finetune['defect'].replace({'True': 1.0, 'False': 0.0}).astype('float32')
finetune['watcher'] = finetune['watcher'].replace({'True': 1.0, 'False': 0.0}).astype('float32')
finetune['victory'] = finetune['victory'].replace({'True': 1.0, 'False': 0.0}).astype('float32')

# Shift down data type sizes - update this later
finetune = finetune.astype('float32')

print('train/test splitting')
# Train test split of the data
ft_train, ft_test = train_test_split(finetune, test_size=0.2)
del finetune

# Print baseline statistics ~4-5% win rate in finetuning
print('-' * 15)
print('underlying victory probabilities')
print('train: ' + str(ft_train['victory'].sum()/len(ft_train)))
print('test: ' + str(ft_test['victory'].sum()/len(ft_test)))

# Setup modeling
# Using a smaller batch size because it tends to perform better in imbalanced data per: https://arxiv.org/pdf/2312.02517
batch_size = 64

# Create data loaders.
# Passing to DataLoader
ft_train = TensorDataset(torch.from_numpy(ft_train.drop('victory', axis=1).to_numpy().astype(np.float32)), torch.from_numpy(ft_train['victory'].to_numpy().astype(np.float32)))
ft_train_loader = DataLoader(ft_train, batch_size=batch_size, shuffle=True)

ft_test = TensorDataset(torch.from_numpy(ft_test.drop('victory', axis=1).to_numpy().astype(np.float32)), torch.from_numpy(ft_test['victory'].to_numpy().astype(np.float32)))
ft_test_loader = DataLoader(ft_test, batch_size=batch_size, shuffle=True)

for X, y in ft_train_loader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

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
            nn.Linear(741, 741),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(741, 741),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(741, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

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
    return test_loss

epochs = 50
loss_tracker = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(ft_train_loader, model, loss_fn, optimizer)
    epoch_test_loss = test(ft_test_loader, model, loss_fn)

    loss_tracker += [epoch_test_loss]

    # Save out the model
    model_path = f'{project_dir}/model_objects/dev/fnn_v8/epoch_{t}.pth'
    model_scripted = torch.jit.script(model) # Export to TorchScript
    model_scripted.save(model_path) # Save
    print(f"Saved PyTorch Model State to {model_path}")

print("Done!")

index_min = np.argmin(loss_tracker)
print(f'Best Model Was Epoch: {index_min + 1} with a loss of {loss_tracker[index_min]}')

# Scores the test set
with torch.no_grad():
    best_model = torch.jit.load(f'{project_dir}/model_objects/dev/fnn_v8/epoch_{index_min}.pth').to(device)
    ft_test = ft_test.to(device)
    pred = best_model(ft_test.float())

ft_test['pred'] = pred
ft_test.to_csv('C:/Users/jjoba/sts/data/test_predictions_fnn_v8.csv', index = False)
