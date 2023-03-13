"""
Recreating the room occupancy experiment
from the MIT liquid time-constant networks paper.

The paper can be found at: https://arxiv.org/abs/2006.04439
The original code from the paper can be found at: https://github.com/raminmh/liquid_time_constant_networks
"""

import numpy as np
import pandas as pd
import time
from sklearn.metrics import accuracy_score

from run_experiment import cut_in_sequences, run_full_experiment


# The following three functions are from the MIT experiments, copied here without alteration
# Except for removing the normalization, which is not necessary for the MaRz process
def read_file(filename):
    df = pd.read_csv(filename)

    data_x = np.stack([
        df['Temperature'].values,
        df['Humidity'].values,
        df['Light'].values,
        df['CO2'].values,
        df['HumidityRatio'].values,
        ], axis=-1)
    data_y = df['Occupancy'].values.astype(np.int32)
    return data_x, data_y


def load_data_from_mit():
    """
    INTENT: Load the dataset from the MIT experiment.

    PRE 1: The dataset is split into three files.

    POST 1: The dataset is re-joined into a single dataset and returned.
    """
    # from MIT code, get the full dataset
    train_x, train_y = read_file("data/occupancy/datatraining.txt")
    test0_x, test0_y = read_file("data/occupancy/datatest.txt")
    test1_x, test1_y = read_file("data/occupancy/datatest2.txt")

    # un-split the data, since MaRz does not need train/test splits
    data_ = np.concatenate((train_x, test0_x, test1_x))
    targets_ = np.concatenate((train_y, test0_y, test1_y))
    data_, targets_ = cut_in_sequences(data_, targets_, 1, 1)  # numbers from person.py: 32, 32//2
    data_ = data_.squeeze(0)  # remove extra dimension from sequences
    targets_ = targets_.reshape(targets_.shape[1], 1)  # make it match for concatenation

    print("data shape =", np.shape(data_), " targets shape =", np.shape(targets_))
    return np.concatenate((data_, targets_), axis=1)


if __name__ == '__main__':
    """
    INTENT: run the occupancy experiment
    
    PRECONDITION 1: the data is in the data directory
    
    POSTCONDITION 1: the dataset is run through MaRz and results for each input are recorded in y_predicted
    POST 2: accuracy score is calculated by comparing with y_actual and results are printed to the console
    POST 3: the time taken in seconds for each step is printed to the console
    """
    loading_timer = time.time()
    occupancy_dataset = load_data_from_mit()

    load_time = time.time() - loading_timer
    print('=' * 20, f"loaded occupancy dataset in {load_time:.2f} seconds", '=' * 20)

    y_actual, y_predicted = run_full_experiment(occupancy_dataset)

    # Calculate accuracy score
    y_predicted = np.rint(np.array(y_predicted))  # convert to binary array to match targets type
    accuracy = accuracy_score(np.array(y_actual), y_predicted)
    print(f"Accuracy score: {accuracy * 100:.2f}%")
    print("MIT Accuracy score: 94.63% ± 0.017")  # from page 6 of the paper
