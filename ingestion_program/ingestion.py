import json
import os
import sys
import time
import numpy as np
import pandas as pd

import argparse

# Default Paths (useful for docker run if no args provided)
input_dir = "/app/dev_phase/input_data/"  # Input data (train.csv, test.csv)
output_dir = "/app/output/"  # For the predictions
program_dir = "/app/program"
submission_dir = "/app/ingested_program"  # The code submitted by participant


def get_data():
    """Load X_train, y_train and X_test from csv files."""
    train = pd.read_csv(os.path.join(input_dir, "train/train_features.csv"))
    X_train = train["SMILES"]

    train_labels = pd.read_csv(os.path.join(input_dir, "train/train_labels.csv"))
    y_train = train_labels["Label"]

    test = pd.read_csv(os.path.join(input_dir, "test/test_features.csv"))
    X_test = test["SMILES"]

    return X_train, y_train, X_test


def print_bar():
    """Display a separator bar."""
    print("-" * 10)


def main():
    """The ingestion program."""
    print_bar()
    print("Ingestion program.")

    from submission import get_model  # Function submitted by the participant

    start = time.time()

    # Read data
    print("Reading data")
    X_train, y_train, X_test = get_data()

    # Initialize model
    print("Initializing the model")
    model = get_model()

    # Train model
    print("Training the model")
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train

    # Make predictions
    print("Making predictions")
    start_test = time.time()
    y_pred = model.predict(X_test)
    test_time = time.time() - start_test

    # Save predictions
    np.savetxt(os.path.join(output_dir, "test_predictions.csv"), y_pred, fmt="%d")

    duration = time.time() - start
    print(
        f"Completed. Total duration: {duration:.2f}s (Train: {train_time:.2f}s, Test: {test_time:.2f}s)"
    )

    with open(os.path.join(output_dir, "metadata.json"), "w+") as f:
        json.dump({"train_time": train_time, "test_time": test_time}, f)

    print("Ingestion program finished. Moving on to scoring.")
    print_bar()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingestion program for codabench")
    parser.add_argument("--data-dir", type=str, default=input_dir)
    parser.add_argument("--output-dir", type=str, default=output_dir)
    parser.add_argument("--program-dir", type=str, default=program_dir)
    parser.add_argument("--submission-dir", type=str, default=submission_dir)
    args, _ = parser.parse_known_args()

    input_dir = args.data_dir
    output_dir = args.output_dir
    program_dir = args.program_dir
    submission_dir = args.submission_dir

    # create output dir
    os.makedirs(output_dir, exist_ok=True)

    sys.path.append(output_dir)
    sys.path.append(program_dir)
    sys.path.append(submission_dir)

    main()
