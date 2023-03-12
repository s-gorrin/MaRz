"""
Recreating the human activity recognition experiment
from the MIT liquid time-constant networks paper.

The paper can be found at: https://arxiv.org/abs/2006.04439
The original code from the paper can be found at: https://github.com/raminmh/liquid_time_constant_networks
"""

import numpy as np
import time
from sklearn.model_selection import train_test_split

# todo: copy and paste in functions, remove normalization, test a bit more
from person import load_crappy_formated_csv, cut_in_sequences
from run_dataset import preprocessing, run_dataset

loading_timer = time.time()
data_, targets_ = load_crappy_formated_csv()
data_, targets_ = cut_in_sequences(data_, targets_, 1, 1)  # numbers from person.py: 32, 32//2
data_ = data_.squeeze(0)  # remove extra dimension from sequences
targets_ = targets_.reshape(targets_.shape[1], 1)  # make it match for concatenation

print("data shape =", np.shape(data_), " targets shape =", np.shape(targets_))
person_dataset = np.concatenate((data_, targets_), axis=1)
# take 15% test split, per MIT paper
unused_train, person_dataset = train_test_split(person_dataset, test_size=0.15, random_state=0)

load_time = time.time() - loading_timer
print('=' * 20, f"loaded person dataset in {load_time:.2f} seconds", '=' * 20)

preprocessing_timer = time.time()
index_table, base_fuzzy = preprocessing(person_dataset)

preprocessing_time = time.time() - preprocessing_timer
print('=' * 20, f"preprocessed dataset in {preprocessing_time:.2f} seconds", '=' * 20)

# display basic target metrics for dataset
t_min = min(person_dataset[:, -1])
t_max = max(person_dataset[:, -1])
print(f"target min/max/range: {t_min}/{t_max}/{t_max - t_min}")
print("Targets are integers 0-6, so outputs should be rounded to nearest int.")

run_timer = time.time()
y_actual, y_predicted = run_dataset(person_dataset, index_table, base_fuzzy,
                                    points=1, close_threshold=0.5, start=0, step=1000)  # NOTICE: BIG STEP HERE

run_time = time.time() - run_timer
print(f"dataset run time was {run_time:.2f} seconds")

# calculate accuracy score by rounding outputs and comparing with inputs
y_predicted = np.rint(np.array(y_predicted))  # round to the nearest integer

n = len(y_predicted)
acc_array = y_actual - y_predicted
num_wrong = np.count_nonzero(acc_array)
perc_correct = 1 - (num_wrong / n)

print(f"MaRz accuracy Score: {perc_correct * 100:.2f}% correct.")
print("Best from MIT paper: 97.26% correct.")
print("LTC method accuracy: 95.67% correct.")
