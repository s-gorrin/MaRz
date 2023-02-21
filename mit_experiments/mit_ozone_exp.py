"""
Recreating the human activity recognition experiment
from the MIT liquid time-constant networks paper.

The paper can be found at: https://arxiv.org/abs/2006.04439
The original code from the paper can be found at: https://github.com/raminmh/liquid_time_constant_networks
"""

import numpy as np
import time
from sklearn.metrics import f1_score

from run_dataset import preprocessing, run_dataset


# The following three functions are from the MIT experiments, copied here without alteration
# Except for removing the normalization, which is not necessary for the MaRz process
def to_float(v):
    if v == "?":
        return 0
    else:
        return float(v)


def load_trace():
    all_x = []
    all_y = []

    with open("data/ozone/eighthr.data", "r") as f:
        miss = 0
        total = 0
        while True:
            line = f.readline()
            if line is None:
                break
            line = line[:-1]
            parts = line.split(',')

            total += 1
            for i in range(1, len(parts) - 1):
                if parts[i] == "?":
                    miss += 1
                    break

            if len(parts) != 74:
                break
            label = int(float(parts[-1]))
            feats = [to_float(parts[i]) for i in range(1, len(parts)-1)]

            all_x.append(np.array(feats))
            all_y.append(label)
    print("Missing features in {} out of {} samples ({:0.2f})".format(miss, total, 100*miss/total))
    print("Read {} lines".format(len(all_x)))
    all_x = np.stack(all_x, axis=0)
    all_y = np.array(all_y)

    print("Imbalance: {:0.2f}%".format(100*np.mean(all_y)))
    # all_x -= np.mean(all_x) #normalize
    # all_x /= np.std(all_x) #normalize

    return all_x, all_y


def cut_in_sequences(x, y, seq_len, inc=1):

    sequences_x = []
    sequences_y = []

    for s in range(0, x.shape[0] - seq_len, inc):
        start = s
        end = start + seq_len
        sequences_x.append(x[start:end])
        sequences_y.append(y[start:end])

    return np.stack(sequences_x,  axis=1), np.stack(sequences_y, axis=1)


def load_data_from_mit():
    """
    INTENT: Load the dataset from the MIT experiment.

    POST 1: The dataset is ready for MaRz usage and returned.
    """
    data_, targets_ = load_trace()
    data_, targets_ = cut_in_sequences(data_, targets_, 1, 1)  # numbers from person.py: 32, 32//2
    data_ = data_.squeeze(0)  # remove extra dimension from sequences
    targets_ = targets_.reshape(targets_.shape[1], 1)  # make it match for concatenation

    print("data shape =", np.shape(data_), " targets shape =", np.shape(targets_))
    return np.concatenate((data_, targets_), axis=1)


if __name__ == '__main__':
    loading_timer = time.time()
    ozone_dataset = load_data_from_mit()

    load_time = time.time() - loading_timer
    print('=' * 20, f"loaded ozone dataset in {load_time:.2f} seconds", '=' * 20)

    preprocessing_timer = time.time()
    index_table, base_fuzzy = preprocessing(ozone_dataset)

    preprocessing_time = time.time() - preprocessing_timer
    print('=' * 20, f"pre-processed dataset in {preprocessing_time:.2f} seconds", '=' * 20)

    run_timer = time.time()
    y_actual, y_predicted = run_dataset(ozone_dataset, index_table, base_fuzzy,
                                        points=1, close_threshold=0.5, verbose=False)

    run_time = time.time() - run_timer
    print(f"dataset run time was {run_time:.2f} seconds")

    # TODO (not code) figure out what they're actually doing with the data to get results
    #  consider asking the MIT team directly for the missing libraries
    # Calculate F1 score
    y_predicted = np.rint(np.array(y_predicted))  # convert to binary array to match targets type
    f1 = f1_score(np.array(y_actual), y_predicted)
    print(f"F1 score: {f1:.6f}")
    print("MIT F1 score: 0.302 ± 0.0155")  # from page 6 of the paper
