import numpy as np
import unittest


def in_range(a_data_point, data_width, min_fuzzy, max_fuzzy):
    """
    INTENT: determine if the input portion of a_data_point is within the range of min_fuzzy to max_fuzzy

    PRECONDITION 1: a_data_point is a list at least as long as a_fuzzy_width and of the same types
    PRECONDITION 2: the first len(an_input) items of a_data_point are input data

    POST_CONDITION 1: a distance of a_fuzzy_width is examined to either side of each number in
        an_input, and the corresponding number in a_data_point is determined to be within that range or not
    POST-CONDITION 2: a boolean indicating whether a_data_point is within the defined fuzzy_width is returned
    """

    for i in range(data_width):
        if a_data_point[i] <= min_fuzzy[i] or a_data_point[i] >= max_fuzzy[i]:
            return False  # if any column is out of range, the whole point is out of range
    return True


def indices_in_range(some_array, lower_bound, upper_bound):
    """
    INTENT: take a numpy array and boundaries, and return the indices within those bounds.

    PRECONDITION 1: some_array is a numpy array with len(lower_bound) + 1 columns
    PRECONDITION 2: the last column of some_array is the targets column

    POST_CONDITION: a true/false array is generated from some_array such that any line in-bounds
        is true, and these are then used to generate a list of indices.

    RETURN: a list of indices that are within the bounds
    """
    tf_array = (some_array[:, :-1] >= lower_bound) & (some_array[:, :-1] <= upper_bound)
    return np.all(tf_array, axis=1).nonzero()[0].tolist()


def get_alpha(an_input, some_data, base_fuzzy_width, num_data_points, max_iterations):
    """
    INTENT: Find an alpha value which modifies a_fuzzy_width such that it creates a hyper-rectangle
        around an_input within some_data which contains num_data_points of records.
        If the exact number is not found within max_iterations loops, return the best result found.

    PRECONDITION 1: some_data is a list of rows of data, consisting of numbers
    PRECONDITION 2: an_input is a list of numbers in the same scope as rows in some_data
    PRECONDITION 3: a_fuzzy_width contains the ranges of values in each column of some_data
    PRECONDITION 4: num_data_points is less than the number of data points in some_data
    PRECONDITION 5: max_iterations is a positive number of times to search before stopping

    POST_CONDITION: current_alpha is a float between 0 and 1 such that it multiplies a_fuzzy_width
        to create a hyper-rectangle within some_data that contains the points at data_indices

    RETURNS: an alpha value that defines a good-sized hyper-rectangle
    RETURNS: a list of length num_data_points of indices within some_data
    """
    if type(some_data) == list:
        some_data = np.array(some_data)
    data_width = len(an_input)

    current_alpha = 0.4  # starting alpha is modified here, slightly faster if this is over
    data_indices = []
    best_low_alpha = 0
    best_high_alpha = 1
    num_iterations = 0

    # terminates because num_iterations begins at 0 and is incremented only
    # this has to be != because above and below num_data_points would both be incorrect
    while len(data_indices) != num_data_points and num_iterations < max_iterations:
        a_fuzzy_width = [n * current_alpha for n in base_fuzzy_width]
        min_fuzzy = [an_input[i] - a_fuzzy_width[i] for i in range(data_width)]
        max_fuzzy = [an_input[i] + a_fuzzy_width[i] for i in range(data_width)]

        # if we already have more than enough data points, don't look in the whole dataset
        # if found indices is more than half the size of the dataset, use numpy instead
        if len(some_data) // 2 > len(data_indices) > num_data_points:
            candidate_indices = [i for i in data_indices if in_range(some_data[i], data_width, min_fuzzy, max_fuzzy)]
        else:  # search entire dataset
            candidate_indices = indices_in_range(some_data, min_fuzzy, max_fuzzy)
            # candidate_indices = [i for i in range(len(some_data))
            #                     if in_range(some_data[i], data_width, min_fuzzy, max_fuzzy)]

        if len(candidate_indices) >= num_data_points:
            data_indices = candidate_indices  # this run is the new best, so save the results
            # if the right number have been found, stop changing alpha
            if len(candidate_indices) != num_data_points:
                best_high_alpha = current_alpha
                current_alpha -= (best_high_alpha - best_low_alpha) / 2
        else:
            best_low_alpha = current_alpha
            current_alpha += (best_high_alpha - best_low_alpha) / 2

        num_iterations += 1

    return current_alpha, data_indices


class GetAlphaTests(unittest.TestCase):

    DELTA = 0.001

    def test_get_alpha(self):

        some_data_1 = [[1, 1, 1],
                       [2, 2, 1],
                       [3, 3, 1],
                       [4, 4, 2],
                       [6, 6, 2],
                       [7, 7, 3],
                       [8, 8, 3],
                       [9, 9, 3]]

        base_fuzzy = [8, 8]
        an_input_1 = [5, 5]

        # easiest test with small data, all in order
        values_1 = get_alpha(an_input_1, np.array(some_data_1), base_fuzzy, 4, 10)
        assert(values_1[0] * 8 < 3)
        assert(values_1[1] == [2, 3, 4, 5])

        some_data_2 = [[1, 9, 1],
                       [2, 8, 1],
                       [3, 7, 1],
                       [4, 6, 2],
                       [6, 4, 2],
                       [7, 3, 3],
                       [8, 2, 3],
                       [9, 1, 3]]

        # same test as 1, but data is inherently non-sortable
        values_2 = get_alpha(an_input_1, some_data_2, base_fuzzy, 4, 10)
        assert(values_2[0] * 8 < 3)
        assert(values_2[1] == [2, 3, 4, 5])

        # same test as 2 but finding fewer points of output
        values_3 = get_alpha(an_input_1, some_data_2, base_fuzzy, 2, 10)
        assert(values_3[0] * 8 < 2)
        assert(values_3[1] == [3, 4])

        some_data_4 = [[3.9, 2.4, 1.4, 0.4, 2.0],
                       [4.0, 2.5, 1.5, 0.5, 2.0],
                       [4.1, 2.6, 1.6, 0.6, 2.0],
                       [4.2, 2.7, 1.7, 0.7, 2.0],
                       [4.3, 2.8, 1.8, 0.8, 2.0],
                       [4.4, 2.9, 1.9, 0.9, 2.0],  # middle below this line
                       [4.6, 3.1, 2.1, 1.1, 2.0],
                       [4.7, 3.2, 2.2, 1.2, 2.0],
                       [4.8, 3.3, 2.3, 1.3, 2.0],
                       [4.9, 3.4, 2.4, 1.4, 2.0],
                       [5.0, 3.5, 2.5, 1.5, 2.0],
                       [5.1, 3.6, 2.6, 1.6, 2.0]]
        an_input_4 = [4.5, 3.0, 2.0, 1.0]
        base_fuzzy_4 = [1.2, 1.2, 1.2, 1.2]

        values_4 = get_alpha(an_input_4, some_data_4, base_fuzzy_4, 4, 10)
        assert(values_4[0] * 1.2 < 0.3)
        assert(values_4[1] == [4, 5, 6, 7])

        an_input_5 = [5.2, 3.7, 2.7, 1.7]

        # edge case: input is after the edge of the dataset
        values_5 = get_alpha(an_input_5, some_data_4, base_fuzzy_4, 4, 10)
        assert(values_5[0] * 1.2 < 0.5)
        assert(values_5[1] == [8, 9, 10, 11])

        # same as some_data_4, but shuffled somewhat
        some_data_6 = [[3.9, 2.4, 1.4, 0.4, 2.0],
                       [4.3, 2.8, 1.8, 0.8, 2.0],
                       [4.4, 2.9, 1.9, 0.9, 2.0],  # middle below this line (in sorted version)
                       [4.6, 3.1, 2.1, 1.1, 2.0],
                       [5.0, 3.5, 2.5, 1.5, 2.0],
                       [4.0, 2.5, 1.5, 0.5, 2.0],
                       [4.9, 3.4, 2.4, 1.4, 2.0],
                       [4.8, 3.3, 2.3, 1.3, 2.0],
                       [4.7, 3.2, 2.2, 1.2, 2.0],
                       [4.1, 2.6, 1.6, 0.6, 2.0],
                       [4.2, 2.7, 1.7, 0.7, 2.0],
                       [5.1, 3.6, 2.6, 1.6, 2.0]]

        values_6 = get_alpha(an_input_4, some_data_6, base_fuzzy_4, 4, 10)
        assert(values_6[0] * 1.2 < 0.5)
        assert(values_6[1] == [1, 2, 3, 8])

        # this shows that the same data will find the same lines for the same input
        # regardless of the order of the data
        for i in range(4):
            assert(some_data_4[values_4[1][i]] == some_data_6[values_6[1][i]])

        # test with real iris data
        some_data_7 = [[5.4, 3.9, 1.7, 0.4, 1.0],
                       [5.5, 2.3, 4.0, 1.3, 2.0],
                       [5.5, 2.4, 3.8, 1.1, 2.0],
                       [5.5, 2.5, 4.0, 1.3, 2.0],
                       [5.5, 2.6, 4.4, 1.2, 2.0],
                       [5.5, 3.5, 1.3, 0.2, 1.0],
                       [5.5, 4.2, 1.4, 0.2, 1.0],
                       [5.6, 2.5, 3.9, 1.1, 2.0],
                       [5.6, 2.7, 4.2, 1.3, 2.0],
                       [5.6, 2.8, 4.9, 2.0, 3.0],
                       [5.6, 2.9, 3.6, 1.3, 2.0],
                       [5.6, 3.0, 4.1, 1.3, 2.0],
                       [5.6, 3.0, 4.5, 1.5, 2.0],
                       [5.7, 2.5, 5.0, 2.0, 3.0],
                       [5.7, 2.8, 4.1, 1.3, 2.0],
                       [5.7, 2.8, 4.5, 1.3, 2.0],
                       [5.7, 2.9, 4.2, 1.3, 2.0],
                       [5.7, 3.0, 4.2, 1.2, 2.0],
                       [5.8, 2.6, 4.0, 1.2, 2.0],
                       [5.8, 2.7, 3.9, 1.2, 2.0],
                       [5.8, 2.7, 4.1, 1.0, 2.0],
                       [5.8, 2.7, 5.1, 1.9, 3.0],
                       [5.8, 2.7, 5.1, 1.9, 3.0],
                       [5.8, 2.8, 5.1, 2.4, 3.0],
                       [5.8, 4.0, 1.2, 0.2, 1.0],
                       [5.9, 3.0, 4.2, 1.5, 2.0]]

        an_input_7 = [5.7, 2.6, 3.5, 1.0]  # target for this input is 2.0
        base_fuzzy_7 = [0.5, 1.9, 3.9, 2.2]

        # some notes about this:
        #   1. with 10 iterations, it finds 8 results
        #       for those 8 indices, all of them are target 2 in the data, meaning it should get a good result
        #   2. with 50 iterations it finds 5 results, all of which are good, and members of the 8-result set
        #   3. with 1000 iterations, it still only finds 5 results, which tells me a box of
        #       exactly four records is not possible
        #   4. with 1000 iterations, it still only takes 0.01 seconds to run all these tests (per unittest)
        #       which I assume is because it only has to look at the full dataset enough times to create
        #       a viable subset (the 8 results), and looking at those 8 results 1000 times is trivial
        #   5. furthermore, any time it creates a smaller subset, it replaces the larger subset,
        #       making subsequent runs even faster (so for most of the 1000 runs, it only checks 5 indices)
        values_7 = get_alpha(an_input_7, some_data_7, base_fuzzy_7, 4, max_iterations=10)
        print(values_7[0], "<- alpha  7  indices ->", values_7[1])
        assert(abs(values_7[0] - 0.2) < self.DELTA)
