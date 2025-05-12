import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, make_scorer, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from pyoperon.sklearn import SymbolicRegressor
from pyoperon import R2, MSE, InfixFormatter, FitLeastSquares, Interpreter
import re

from src.preprocess_data.preprocess_data import load_Sajjad
from sympy import parse_expr
import matplotlib.pyplot as plt
from copy import deepcopy
from definitions import ROOT_DIR
from src.preprocess_data.config_load_dataset import ConfigLoadData

# Define the 3-layer MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, dtype=float)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size,dtype=float)
        self.fc3 = nn.Linear(hidden_size, output_size, dtype=float)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

# Hyperparameters

hidden_size = 50  # Example hidden layer size
output_size = 1   # Example output size
learning_rate = 0.001
batch_size = 32
num_epochs = 1000


parser = ConfigLoadData.arguments_parser()
args = parser.parse_args()

#D_test =  load_Sajjad(args,f'{ROOT_DIR}/data/Sajjad/full_dataset500_RoboSci_V3_no-defect_97.csv')
D_train = load_Sajjad(args, f'{ROOT_DIR}/data/Sajjad/full_dataset500_RoboSci_V3_no-defect_217.csv')


x, y = D_train.loc[:, args.features].to_numpy(), D_train.loc[:,['y']].to_numpy()
#X_test, y_test = D_test.loc[:, args.features].to_numpy(), D_test.loc[:,['y']].to_numpy()
x=torch.tensor(x)
y=torch.tensor(y[:,0]) * 1000

# Create DataLoader
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, loss function, and optimizer
model = MLP(x.shape[1], hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

string = (f"{int(D_train.loc[0, 'Video ID'])} "
          f"{round(np.rad2deg(D_train.loc[0, 'tilt_angle']))}° "
          f"{D_train.loc[0, 'excel_name']}")

print("Training complete!")
predict = model(x).detach().numpy()
fig, (ax1) = plt.subplots(figsize=(7, 5),nrows=1, sharex=True, sharey=True)
ax1.set_title(string)
ax1.scatter(range(len(predict)), D_train.loc[:,['y']].to_numpy()* 1000, label='true', s=1)
ax1.scatter(range(len(predict)), predict, label='prediction NN', s=1)
#ax1.bar(range(len(predict)), (D_train.loc[:,['y']].to_numpy() - predict)[:,0], label='difference' )
ax1.legend(loc='upper right')
ax1.set_ylabel('Friction Force')
plt.show()