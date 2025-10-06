import argparse
import csv
import os
from datetime import datetime

import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from data import PolarisDataset
from models import GINModel
from StratifiedTanimotoSplit import StratifiedTanimotoSplit
from utils import ScaffoldKFold
from greedy_random_split import GreedyKFold

def main(params: dict):
    torch.manual_seed(params["seed"])
    torch.set_num_threads(1)

    # Load polaris dataset for a specific endpoint ("MLM", "HLM", etc.)
    # Split according to predefined Time split: -> train_dataset, test_dataset
    train_dataset = PolarisDataset(
        root="./dataset", task=params["task"], train=True, force_reload=False
    )
    test_dataset = PolarisDataset(
        root="./dataset", task=params["task"], train=False, force_reload=False
    )

    # Split train dataset into K-Fold Cross validation according to: {Stratified, Scaffold, STS}
    # -> train_fold, valid_fold
    smiles = train_dataset.smiles
    labels = train_dataset.y.view(-1).tolist()

    # Initialize model and loss_fn
    model = GINModel(
        hidden_channels=params["hidden_channels"],
        out_channels=64,
        num_layers=params["num_layers"],
        dropout=params["dropout"],
        encoding_dim=8,
    )
    loss_fn = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=params["lr"], weight_decay=1e-4)

    match params["split_method"]:
        case "stratified":
            avg_final_valid_loss = stratified_kfold_cross_validation(
                params, train_dataset, smiles, labels, model, loss_fn
            )
        case "scaffold":
            avg_final_valid_loss = scaffold_kfold_cross_validation(
                params, train_dataset, smiles, None, model, loss_fn, optimizer
            )
        case "sts":
            avg_final_valid_loss = sts_kfold_cross_validation(
                params, train_dataset, smiles, labels, model, loss_fn
            )
            print("Performing KFold STS")
            
        case "greedy":
            print("Performing Greedy Splitting")
            avg_final_valid_loss = greedy_kfold_cross_validation(
                params, train_dataset, smiles, labels, model, loss_fn, optimizer)

        case _:
            raise ValueError(f"Unknown splitting method: {params['split_method']}")

    # Add results to params dictionary:
    params["avg_final_val_loss"] = avg_final_valid_loss

    # Reset the model parameters and the optimizer
    model.reset_parameters()
    optimizer = Adam(model.parameters(), lr=params["lr"], weight_decay=1e-4)

    # Retrain the model on the entire train dataset
    train_dataloader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
    for epoch in range(params["epochs"]):
        train_loop(train_dataloader, model, loss_fn, optimizer)

    # Evaluate the final model on the test dataset
    test_dataloader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False)
    test_loss = test_loop(test_dataloader, model, loss_fn)

    params["test_loss"] = test_loss

    # Save each run in a separate file
    os.makedirs("results", exist_ok=True)
    output_path = f"results/{datetime.now().isoformat()}.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(params.keys())
        writer.writerow(params.values())


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()

    for data in dataloader:
        out = model(data)
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test_loop(dataloader, model, loss_fn):
    model.eval()
    loss_list = []

    with torch.no_grad():
        for data in dataloader:
            out = model(data)
            loss = loss_fn(out, data.y)
            loss_list.append(loss.item())

    return sum(loss_list) / len(loss_list)


def stratified_kfold_cross_validation(params, train_dataset, smiles, labels, model, loss_fn):
    y_binned = pd.qcut(labels, q=10, labels=False)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=params["seed"])

    per_fold_final = []
    for train_idx, valid_idx in skf.split(smiles, y_binned):
        # Reset the model parameters and the optimizer
        model.reset_parameters()
        optimizer = Adam(model.parameters(), lr=params["lr"], weight_decay=1e-4)

        train_fold = train_dataset[train_idx]
        valid_fold = train_dataset[valid_idx]

        train_fold_dataloader = DataLoader(
            train_fold, batch_size=params["batch_size"], shuffle=True
        )
        valid_fold_dataloader = DataLoader(
            valid_fold, batch_size=params["batch_size"], shuffle=False
        )

        last_val = None

        for epoch in range(params["epochs"]):
            train_loop(train_fold_dataloader, model, loss_fn, optimizer)
            valid_loss = test_loop(valid_fold_dataloader, model, loss_fn)
            last_val = valid_loss

        per_fold_final.append(last_val)

    avg_final_valid_loss = sum(per_fold_final) / len(per_fold_final)

    return avg_final_valid_loss


def scaffold_kfold_cross_validation(
    params, train_dataset, smiles, labels, model, loss_fn, optimizer
):
    skf = ScaffoldKFold(n_splits=5, shuffle=True, random_state=params["seed"])

    per_fold_final = []
    for train_idx, valid_idx in skf.split(smiles):
        # Reset the model parameters and the optimizer
        model.reset_parameters()
        optimizer = Adam(model.parameters(), lr=params["lr"], weight_decay=1e-4)

        train_fold = train_dataset[train_idx]
        valid_fold = train_dataset[valid_idx]

        train_fold_dataloader = DataLoader(
            train_fold, batch_size=params["batch_size"], shuffle=True
        )
        valid_fold_dataloader = DataLoader(
            valid_fold, batch_size=params["batch_size"], shuffle=False
        )

        last_val = None

        for epoch in range(params["epochs"]):
            train_loop(train_fold_dataloader, model, loss_fn, optimizer)
            valid_loss = test_loop(valid_fold_dataloader, model, loss_fn)
            last_val = valid_loss

        per_fold_final.append(last_val)

    avg_final_valid_loss = sum(per_fold_final) / len(per_fold_final)

    return avg_final_valid_loss


def sts_kfold_cross_validation(params, train_dataset, smiles, labels, model, loss_fn):
    labels = torch.tensor(labels)
    skf = StratifiedTanimotoSplit(train_dataset)

    per_fold_final = []
    for train_idx, valid_idx in skf.split(target=labels, smiles=smiles):
        # Reset the model parameters and the optimizer
        model.reset_parameters()
        optimizer = Adam(model.parameters(), lr=params["lr"], weight_decay=1e-4)

        train_fold = train_dataset[train_idx]
        valid_fold = train_dataset[valid_idx]

        train_fold_dataloader = DataLoader(
            train_fold, batch_size=params["batch_size"], shuffle=True
        )
        valid_fold_dataloader = DataLoader(
            valid_fold, batch_size=params["batch_size"], shuffle=False
        )

        last_val = None

        for epoch in range(params["epochs"]):
            train_loop(train_fold_dataloader, model, loss_fn, optimizer)
            valid_loss = test_loop(valid_fold_dataloader, model, loss_fn)
            last_val = valid_loss

        per_fold_final.append(last_val)

    avg_final_valid_loss = sum(per_fold_final) / len(per_fold_final)

    return avg_final_valid_loss

def greedy_kfold_cross_validation(
    params, train_dataset, smiles, labels, model, loss_fn, optimizer
):
    skf = GreedyKFold(n_splits=5, random_state=params["seed"])
    avg_final_valid_loss = 0
    for epoch in range(params["epochs"]):
        # print(f"Epoch {epoch + 1}\n-------------------------------")
        valid_loss_list = []
        for train_idx, valid_idx in skf.split(smiles, labels):
            print(f"train_idx: {train_idx}, valid_idx: {valid_idx}")
            train_fold = train_dataset[train_idx]
            valid_fold = train_dataset[valid_idx]
            train_fold_dataloader = DataLoader(
                train_fold, batch_size=params["batch_size"], shuffle=True
            )
            valid_fold_dataloader = DataLoader(
                valid_fold, batch_size=params["batch_size"], shuffle=False
            )
            train_loop(train_fold_dataloader, model, loss_fn, optimizer)
            valid_loss = test_loop(valid_fold_dataloader, model, loss_fn)
            valid_loss_list.append(valid_loss)
        print(f"Valid losses: {valid_loss_list}")
        print(f"Average valid loss: {sum(valid_loss_list) / len(valid_loss_list)}\n")
        if epoch == params["epochs"] - 1:
            avg_final_valid_loss = sum(valid_loss_list) / len(valid_loss_list)
    return avg_final_valid_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass in the parameters.")

    parser.add_argument("--task", help="Task name", default="MLM")
    parser.add_argument("--batch_size", help="Batch Size", default=16)
    parser.add_argument("--dropout", help="Batch Size", default=0.1)
    parser.add_argument("--epochs", help="Epochs", default=10)
    parser.add_argument("--hidden_channels", help="Epochs", default=32)
    parser.add_argument("--num_layers", help="Epochs", default=3)
    parser.add_argument("--lr", help="Learning rate", default=0.001)
    parser.add_argument("--seed", help="Seed", default=42)
    parser.add_argument("--split_method", help="Splitting method", default="scaffold")

    input_args = parser.parse_args()
    input_args_dict = vars(input_args)

    main(input_args_dict)
