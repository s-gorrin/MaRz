import numpy as np
import unittest


def generate_index_table(some_data):
    """
    INTENT: take in a dataset and generate a table of indices in which each column contains
        the indices of the data set in order as sorted by that column

    PRE 1: some_data is a dataset of shape (x, y) where x is the number of rows and y is the number of columns

    POST 1: an ID column is added to some_data to preserve original index order
    POST 2: each column is stable sorted and the new order of the ID column is added to the output array

    RETURN: a numpy array of the same size as some_data with each column containing the IDs
            of some_data in the order attained by sorting on the given column
    """
    length = len(some_data)
    width = len(some_data[0])

    ids = np.arange(length).reshape((length, 1))  # generate IDs column and shape it for appending to some_data
    some_data = np.concatenate((some_data, ids), axis=1)

    index_table = np.empty((length, width), int)  # create an empty array of some_data.shape to store ints

    for column in range(width):  # this will exclude the new IDs column, but include any targets
        index_table[:, column] = some_data[some_data[:, column].argsort(kind='stable')][:, -1]

    return index_table


def get_base_fuzzy(some_data):
    """
    INTENT: generate a list of column ranges for a dataset

    PRE 1: some_data is a dataset

    POST 1: a list of column ranges is obtained by subtracting the min of each column from the max

    RETURN: the base fuzzy ranges for the dataset

    KNOWN ISSUE: if a column has all one value in it, the base fuzzy for that column will be 0.
        This would cause a division by zero error downstream, so those 0s are replaced with 0.000001.
    """
    min_per_col = np.min(some_data, 0)[:-1]
    max_per_col = np.max(some_data, 0)[:-1]
    base_fuzzy = max_per_col - min_per_col
    base_fuzzy[base_fuzzy == 0] = 0.000001  # convert 0s in the base fuzzy to very small numbers
    return base_fuzzy


class Tests(unittest.TestCase):
    data_set_1 = [[1, 1, 1],
                  [2, 2, 1],
                  [3, 3, 1],
                  [4, 4, 2],
                  [6, 6, 2],
                  [7, 7, 3],
                  [8, 8, 3],
                  [9, 9, 3]]

    data_set_2 = [[1, 9, 1],
                  [2, 8, 1],
                  [3, 7, 1],
                  [4, 6, 2],
                  [6, 4, 2],
                  [7, 3, 3],
                  [8, 2, 3],
                  [9, 1, 3]]

    data_set_3 = [[1, 5, 1],
                  [2, 5, 1],
                  [3, 5, 1],
                  [4, 5, 2],
                  [6, 5, 2],
                  [7, 5, 3],
                  [8, 5, 3],
                  [9, 5, 3]]

    def test_generate_index_table(self):
        output_1 = generate_index_table(self.data_set_1)
        assert(output_1[0][0] == 0)
        assert(output_1[7][1] == 7)

        output_2 = generate_index_table(self.data_set_2)
        assert(output_2[0][0] == 0)
        assert(output_2[7][0] == 7)
        assert(output_2[0][1] == 7)
        assert(output_2[7][1] == 0)

        assert(np.array(self.data_set_1).shape == (8, 3))  # show that the some_data is not changed

        output_3 = generate_index_table(self.data_set_3)
        assert(output_3.shape[1] == 3)

    def test_get_base_fuzzy(self):
        output_1 = get_base_fuzzy(self.data_set_1)
        output_2 = get_base_fuzzy(self.data_set_2)

        assert(output_1[0] == 8)
        assert(output_1[1] == 8)
        assert(output_2[0] == 8)
        assert(output_2[1] == 8)
