import argparse

import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from data import PolarisDataset
from models import GINModel


def main(params: dict):
    # Load polaris dataset for a specific endpoint ("MLM", "HLM", etc.)
    # Split according to predefined Time split: -> train_dataset, test_dataset
    train_dataset = PolarisDataset(root="./dataset", task="MLM", train=True, force_reload=True)
    test_dataset = PolarisDataset(root="./dataset", task="MLM", train=False, force_reload=True)

    print(train_dataset[0])
    print(test_dataset[0])

    # Split train dataset into K-Fold Cross validation according to: {Stratified, Scaffold, STS}
    # -> train_fold, valid_fold
    smiles = train_dataset.smiles
    labels = train_dataset.y.view(-1).tolist()

    y_binned = pd.qcut(labels, q=10, labels=False)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize model and loss_fn
    model = GINModel(hidden_channels=32, out_channels=64, num_layers=3, dropout=0.1, encoding_dim=8)
    loss_fn = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    for train_idx, valid_idx in skf.split(smiles, y_binned):
        train_fold = train_dataset[train_idx]
        valid_fold = train_dataset[valid_idx]

        train_fold_dataloader = DataLoader(train_fold, batch_size=32, shuffle=True)
        valid_fold_dataloader = DataLoader(valid_fold, batch_size=32, shuffle=False)

        train_loop(train_fold_dataloader, model, loss_fn, optimizer)
        test_loop(valid_fold_dataloader, model, loss_fn)

    # Train the GIN-E model with a set of hyperparameters on the train and valid folds
    # Take the average value accross all MAE validations and choose the best one

    # Choose the hyperparameters with the lowest average MAE and retrain on the entire train dataset
    # Evaluate on the test dataset, and get an MAE score


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()

    epoch_loss = 0

    for data in dataloader:
        out = model(data)
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()

    return epoch_loss


def test_loop(dataloader, model, loss_fn):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for data in dataloader:
            out = model(data)
            test_loss += loss_fn(out, data.y)

    return test_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass in the parameters.")

    parser.add_argument("--task", help="Task name", default="MLM")

    input_args = parser.parse_args()
    input_args_dict = vars(input_args)

    main(input_args_dict)
