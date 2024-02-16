import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from IVIMNET.deep import Net, learn_IVIM, predict_IVIM, checkarg
from IVIMNET.fitting_algorithms import fit_dats
from hyperparams import hyperparams as hp

class MRIDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.file_names = [f for f in os.listdir(data_folder) if f.endswith('.nii.gz')]
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        # Load and preprocess data here
        # Example: load NIfTI file, preprocess, and return as tensor
        # data = preprocess_function(os.path.join(self.data_folder, file_name))
        # if self.transform:
        #     data = self.transform(data)
        return torch.randn(1, 10), torch.tensor(1)  # Dummy data, replace with actual preprocessed data

# Set up hyperparameters
arg = hp()
arg = checkarg(arg)

# Set up data paths
data_folder = "./data/PDAC/"

# Set up data loaders
dataset = MRIDataset(data_folder)
train_indices, val_test_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
val_indices, test_indices = train_test_split(val_test_indices, test_size=0.5, random_state=42)

train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

train_loader = DataLoader(dataset, batch_size=arg.train_pars.batch_size, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=arg.train_pars.batch_size, sampler=val_sampler)
test_loader = DataLoader(dataset, batch_size=arg.train_pars.batch_size, sampler=test_sampler)

# Set up the model
model = Net(input_size=10, hidden_size=20, output_size=1)  # Example model, replace with your model

# Set up loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=arg.train_pars.lr)

# Training loop
for epoch in range(arg.train_pars.num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation loop
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for inputs, labels in val_loader:
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
        val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}/{arg.train_pars.num_epochs}, Loss: {loss.item()}, Val Loss: {val_loss}")

# Testing loop
model.eval()
with torch.no_grad():
    test_loss = 0.0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        test_loss += criterion(outputs, labels).item()
    test_loss /= len(test_loader)

print(f"Test Loss: {test_loss}")
