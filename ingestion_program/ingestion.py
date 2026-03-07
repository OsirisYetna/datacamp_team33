import json
import os
import sys
import time
import numpy as np
import pandas as pd

# Paths
input_dir    = '/app/input_data/'      # Input data (train.csv, test.csv)
output_dir   = '/app/output/'          # For the predictions
program_dir  = '/app/program'
submission_dir = '/app/ingested_program'  # The code submitted by participant

sys.path.append(output_dir)
sys.path.append(program_dir)
sys.path.append(submission_dir)


def get_data():
    """Load X_train, y_train and X_test from csv files."""
    train = pd.read_csv(os.path.join(input_dir, 'train.csv'))
    X_train = train['SMILES']

    train_labels = pd.read_csv(os.path.join(input_dir, 'train_labels.csv'))
    y_train = train_labels['Label']

    test = pd.read_csv(os.path.join(input_dir, 'test.csv'))
    X_test = test['SMILES']

    return X_train, y_train, X_test


def print_bar():
    """Display a separator bar."""
    print('-' * 10)


def main():
    """The ingestion program."""
    print_bar()
    print('Ingestion program.')

    from submission import get_model  # Function submitted by the participant

    start = time.time()

    # Read data
    print('Reading data')
    X_train, y_train, X_test = get_data()

    # Initialize model
    print('Initializing the model')
    model = get_model()

    # Train model
    print('Training the model')
    model.fit(X_train, y_train)

    # Make predictions
    print('Making predictions')
    y_pred = model.predict(X_test)

    # Save predictions
    np.savetxt(os.path.join(output_dir, 'test_predictions.csv'), y_pred, fmt='%d')

    duration = time.time() - start
    print(f'Completed. Total duration: {duration:.2f}s')

    with open(os.path.join(output_dir, 'metadata.json'), 'w+') as f:
        json.dump({'duration': duration}, f)

    print('Ingestion program finished. Moving on to scoring.')
    print_bar()


if __name__ == '__main__':
    main()