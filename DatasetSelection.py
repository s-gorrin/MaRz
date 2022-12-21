from sklearn import datasets

from run_dataset import run_dataset, preprocessing
from dataset_preprocessing import *

import matplotlib.pyplot as plt
import seaborn as sns


# A wrapper class to load and format toy datasets from sklearn

# Special note: defaults to breast cancer data set

class DatasetSelection:

    base_fuzzy = []

    def __init__(self, a_ds_name):
        """
        POSTCONDITION 1: self.dataset is the data set defined by a_ds_name for
            diabetes, iris, digits, or wine, otherwise it is breast cancer.

        POST 2: Datasets are in the form [[input list], [output list]]

        POST 3: as per post-condition of calculate_base_fuzzy()
        """

        # ---- POST 1
        if a_ds_name == 'diabetes':
            data_set = datasets.load_diabetes()
        elif a_ds_name == 'iris':
            data_set = datasets.load_iris()
        elif a_ds_name == 'digits':
            data_set = datasets.load_digits()
        elif a_ds_name == 'wine':
            data_set = datasets.load_wine()
        else:
            data_set = datasets.load_breast_cancer()

        # ---- POST 2
        data_ = data_set.data
        targets_ = data_set.target.reshape(len(data_set.target), 1)  # output as column
        self.dataset = np.concatenate((data_, targets_), axis=1)

        # ---- POST 3
        self.calculate_base_fuzzy()

    def calculate_base_fuzzy(self):
        # POSTCONDITION: self.base_fuzzy = ranges of the features in self.dataset

        ds_min = np.min(self.dataset, 0)[:-1]
        ds_max = np.max(self.dataset, 0)[:-1]
        self.base_fuzzy = ds_max - ds_min  # [ds_max[i] - ds_min[i] for i in range(len(ds_min))]


if __name__ == '__main__':

    # POSTCONDITION 1 (MaRz Ran): Marz was run on 8x8 sklearn digits (Named miniMNIST here)
    # POST 2 (Display): The resulting graph is on the monitor of actual vs. predicted digits

    # ---- POST 1 (MaRz Ran)
    dataset = DatasetSelection('digits').dataset
    index_table, input_ranges = preprocessing(dataset)
    y_actual, y_predicted = run_dataset(
        dataset, index_table, input_ranges, points=1, close_threshold=0.1)

    # ---- POST 2 (Display)

    resid = sns.regplot(x=y_actual, y=y_predicted,
                        scatter_kws={'alpha': 0.2},
                        line_kws={'color': 'red', 'alpha': 0.5})
    resid.set(xlabel='Actual Digit', ylabel='Predicted Digit')
    resid.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # digits numbers
    resid.set(title='Mini-MNIST Identification Accuracy')
    plt.show()
