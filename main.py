def main():
    print("Hello World")
    # Load polaris dataset for a specific endpoint ("MLM", "HLM", etc.)
    # Split according to predefined Time split: -> train_dataset, test_dataset

    # Split train dataset into K-Fold Cross validation according to: {Stratified, Scaffold, STS}
    # -> train_fold, valid_fold

    # Train the GIN-E model with a set of hyperparameters on the train and valid folds
    # Take the average value accross all MAE validations and choose the best one

    # Choose the hyperparameters with the lowest average MAE and retrain on the entire train dataset
    # Evaluate on the test dataset, and get an MAE score


if __name__ == "__main__":
    main()
