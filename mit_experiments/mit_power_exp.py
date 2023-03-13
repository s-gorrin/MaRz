"""
Recreating the power consumption experiment
from the MIT liquid time-constant networks paper.

The paper can be found at: https://arxiv.org/abs/2006.04439
The original code from the paper can be found at: https://github.com/raminmh/liquid_time_constant_networks
"""

import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from run_dataset import preprocessing, run_dataset
from get_alpha_sorted import get_alpha
from marz_get_output import get_output


# The following three functions are from the MIT experiments, copied here without alteration
# Except for removing the normalization, which is not necessary for the MaRz process
def convert_to_floats(feature_col, memory):
    for i in range(len(feature_col)):
        if feature_col[i] == "?" or feature_col[i] == "\n":
            feature_col[i] = memory[i]
        else:
            feature_col[i] = float(feature_col[i])
            memory[i] = feature_col[i]
    return feature_col, memory


def load_crappy_formatted_csv():

    all_x = []
    with open("data/power/household_power_consumption.txt", "r") as f:
        lineno = -1
        memory = [i for i in range(7)]
        for line in f:
            lineno += 1
            if lineno == 0:
                continue
            arr = line.split(";")
            if len(arr) < 8:
                continue
            feature_col = arr[2:]
            feature_col, memory = convert_to_floats(feature_col, memory)
            all_x.append(np.array(feature_col, dtype=np.float32))

    all_x = np.stack(all_x, axis=0)
    # all_x -= np.mean(all_x,axis=0) #normalize
    # all_x /= np.std(all_x,axis=0) #normalize

    all_y = all_x[:, 0].reshape([-1, 1])
    all_x = all_x[:, 1:]

    return all_x, all_y


def cut_in_sequences(x, y, seq_len, inc=1):

    sequences_x = []
    sequences_y = []

    for s in range(0, x.shape[0] - seq_len, inc):
        start = s
        end = start+seq_len
        sequences_x.append(x[start:end])
        sequences_y.append(y[start:end])

    return np.stack(sequences_x, axis=1), np.stack(sequences_y, axis=1)


def load_data_from_mit():
    """
    INTENT: Load the dataset from the MIT experiment.

    POST 1: The dataset is ready for MaRz usage and returned.
    """
    data_, targets_ = load_crappy_formatted_csv()
    data_, targets_ = cut_in_sequences(data_, targets_, 1, 1)  # numbers from person.py: 32, 32//2
    data_ = data_.squeeze(0)  # remove extra dimension from sequences
    targets_ = targets_.reshape(targets_.shape[1], 1)  # make it match for concatenation

    print("data shape =", np.shape(data_), " targets shape =", np.shape(targets_))
    return np.concatenate((data_, targets_), axis=1)


if __name__ == '__main__':
    loading_timer = time.time()
    power_dataset = load_data_from_mit()
    unused_train, power_dataset = train_test_split(power_dataset, test_size=.2, random_state=0)

    load_time = time.time() - loading_timer
    print('=' * 20, f"loaded power dataset in {load_time:.2f} seconds", '=' * 20)

    preprocessing_timer = time.time()
    index_table, base_fuzzy = preprocessing(power_dataset)

    preprocessing_time = time.time() - preprocessing_timer
    print('=' * 20, f"preprocessed dataset in {preprocessing_time:.2f} seconds", '=' * 20)

    # display basic target metrics for dataset
    t_min = min(power_dataset[:, -1])
    t_max = max(power_dataset[:, -1])
    print(f"target min/max/range: {t_min:.4f}/{t_max:.4f}/{t_max - t_min:.4f}")

    run_timer = time.time()

    # run the test portion of the dataset. This is very time-consuming due to the size of the data
    y_actual, y_predicted = run_dataset(power_dataset, index_table, base_fuzzy, step=100,
                                        points=1, close_threshold=0.5, verbose=False)
    """
    query_input = power_dataset[56]  # arbitrarily chosen line for input
    alpha, indices = get_alpha(query_input[:-1], power_dataset, index_table, base_fuzzy, 1, 10)
    query_output = get_output(query_input[:-1], power_dataset, base_fuzzy*alpha, indices)
    """

    run_time = time.time() - run_timer
    print('=' * 20, f"run time was {run_time:.2f} seconds", '=' * 20)
    # print(f"target: {query_input[-1]}; output: {query_output}")

    # TODO: write output lists to a file as csv or something for reproducibility, since they take so long to make
    squared_error = mean_squared_error(y_actual, y_predicted)
    print(f"Mean squared error on very small sample of data: {squared_error:.3f}")
    print("Best squared error from paper: 0.586 ± 0.003")
    print("LTC squared error from paper: 0.642 ± 0.021")
