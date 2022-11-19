"""
Data source: https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise

Citation:
    Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [https://archive.ics.uci.edu/ml].
    Irvine, CA: University of California, School of Information and Computer Science.
"""

import pandas as pd
from run_dataset import run_dataset, preprocessing


df = pd.read_csv('airfoil_self_noise.dat', sep='\s+',
                 usecols=['Frequency', 'Attack', 'Chord', 'Velocity', 'Thickness', 'Decibels'])

pd.set_option('display.max_columns', None)
print(df.describe().transpose())

print("\ntargets range:", df['Decibels'].max() - df['Decibels'].min())

data = df.to_numpy()

index_table, base_fuzzy = preprocessing(data)

# observation: close_threshold is effectively making t/ct bins, where t = targets.max - targets.min
run_dataset(data, index_table, base_fuzzy, points=2, close_threshold=4)  # this threshold is likely too high
