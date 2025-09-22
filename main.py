import argparse

from data import PolarisDataset


def main(params: dict):
    pass
    # Load polaris dataset for a specific endpoint ("MLM", "HLM", etc.)
    # Split according to predefined Time split: -> train_dataset, test_dataset
    train_dataset = PolarisDataset(root="./dataset", task="MLM", train=True, force_reload=True)
    test_dataset = PolarisDataset(root="./dataset", task="MLM", train=False, force_reload=True)

    print(train_dataset[0])
    print(test_dataset[0])

    # Split train dataset into K-Fold Cross validation according to: {Stratified, Scaffold, STS}
    # -> train_fold, valid_fold

    # Train the GIN-E model with a set of hyperparameters on the train and valid folds
    # Take the average value accross all MAE validations and choose the best one

    # Choose the hyperparameters with the lowest average MAE and retrain on the entire train dataset
    # Evaluate on the test dataset, and get an MAE score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass in the parameters.")

    parser.add_argument("--task", help="Task name", default="MLM")

    input_args = parser.parse_args()
    input_args_dict = vars(input_args)

    main(input_args_dict)
