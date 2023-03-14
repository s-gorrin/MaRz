# common code to run MIT experiments with MaRz

import numpy as np
from sklearn.model_selection import train_test_split
import time

from run_dataset import preprocessing, run_dataset


# unmodified sequencing code from MIT to make running data more convenient
def cut_in_sequences(x, y, seq_len, inc=1):

    sequences_x = []
    sequences_y = []

    for s in range(0, x.shape[0] - seq_len, inc):
        start = s
        end = start + seq_len
        sequences_x.append(x[start:end])
        sequences_y.append(y[start:end])

    return np.stack(sequences_x, axis=1), np.stack(sequences_y, axis=1)


def run_full_experiment(some_data, split=False, step=1):
    """
    INTENT: run and time an experiment based on the MIT liquid experiments

    PRECONDITION 1: some_data is a dataset formatted for MaRz with targets in the last column
    PRE 2: split is True if the data should be reduced to a 20% training set
    PRE 3: step indicates whether data should be skipped when running
        for step=1, run every line
        for step=2, run every other line
        for step=100, run every 100 lines

    POSTCONDITION 1: the number of seconds taken to preprocess and run the dataset are printed to the console
    POST 2: two parallel lists are returned, first the actual targets from the data and second MaRz predictions
        the returned lists only contain targets/outputs for lines which were run, so if step=2,
            then the returned lists are half the length of the dataset as processed
    """
    if split:
        unused_train, some_data = train_test_split(some_data, test_size=.2, random_state=0)

    preprocessing_timer = time.time()
    index_table, base_fuzzy = preprocessing(some_data)

    preprocessing_time = time.time() - preprocessing_timer
    print('=' * 20, f"pre-processed dataset in {preprocessing_time:.2f} seconds", '=' * 20)

    run_timer = time.time()
    y_actual, y_predicted = run_dataset(some_data, index_table, base_fuzzy, points=1,
                                        close_threshold=0.5, step=step, verbose=False)

    run_time = time.time() - run_timer
    print(f"dataset run time was {run_time:.2f} seconds")

    return y_actual, y_predicted

