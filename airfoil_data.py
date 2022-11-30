"""
Data source: https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise

Citation:
    Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [https://archive.ics.uci.edu/ml].
    Irvine, CA: University of California, School of Information and Computer Science.

Compare R-Square Value with those on page 10 of this paper:
    https://www.sciencedirect.com/science/article/pii/S1877050915000654
"""

import pandas as pd
import time
from sklearn.metrics import r2_score, mean_squared_error

from run_dataset import run_dataset, preprocessing


# read data from file
df = pd.read_csv('airfoil_self_noise.dat', sep='\s+',
                 usecols=['Frequency', 'Attack', 'Chord', 'Velocity', 'Thickness', 'Decibels'])

# display information
pd.set_option('display.max_columns', None)
print(df.describe().transpose())
print("\ntargets range:", df['Decibels'].max() - df['Decibels'].min())

data = df.to_numpy()

start = time.time()
index_table, base_fuzzy = preprocessing(data)

y_actual, y_predicted = run_dataset(data, index_table, base_fuzzy, points=2, close_threshold=2)
end = time.time()

# print results
print(f"\nComputation time (seconds): {end - start:.4f}")
r_square = r2_score(y_actual, y_predicted)
print(f"R-Square Value: {r_square:.5f}")
print(f"Mean Squared Error: {mean_squared_error(y_actual, y_predicted):.5f}")
