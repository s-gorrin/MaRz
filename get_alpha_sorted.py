from dataset_preprocessing import *


def get_alpha(an_input, some_data, index_table, base_fuzzy, num_data_points, max_iterations=10):
    """
    INTENT: use binary search methods to quickly find the hyper-rectangle of some_data which contains
        an_input and num_data_points data points, as defined by an alpha value which multiplies base_fuzzy

    PRE 1: an_input is the size of one row of some_data, containing the same type of data
    PRE 2: index_table is a look-up table of indices of some_data, such that each column contains the indices of
        some_data in the order they would be in if some_data were sorted by that column
    PRE 3: num_data_points is an integer less than the number of rows in some_data
    PRE 4: base_fuzzy is a list of the differences between max and min of each column of some data
    PRE 5: max_iterations is an integer greater than 0

    POST 1: the index_table is used to enable binary search of the dataset for each parameter of an_input
    POST 2: the dataset is searched per column using numpy
    POST 3: the found indices are intersected to create a list of indices within the hyperbox

    RETURN: the alpha value that was found, and the list of indices in the hyper-rectangle defined by alpha
    """
    data_width = len(an_input)

    # this is mostly for testing since all real data should be numpy arrays
    if type(some_data) is not np.ndarray:
        some_data = np.array(some_data)

    current_alpha = 0.1  # starting alpha is modified here; 0.07 seems to be faster, but 0.1 more precise
    best_low_alpha, best_high_alpha = 0, 1
    data_indices = []
    num_iterations = 0

    # terminates because num_iterations begins at 0 and is incremented only
    while len(data_indices) != num_data_points and num_iterations < max_iterations:
        a_fuzzy_width = base_fuzzy * current_alpha
        min_fuzzy = an_input - a_fuzzy_width
        max_fuzzy = an_input + a_fuzzy_width
        candidate_indices = np.array([])

        for c in range(data_width):
            # do the binary search for the range for this column c
            table_low = np.searchsorted(some_data[:, c], min_fuzzy[c], sorter=index_table[:, c])
            table_high = np.searchsorted(some_data[:, c], max_fuzzy[c], sorter=index_table[:, c])

            # if column has all the same value, include the whole column
            if table_high == 0:
                table_high = len(some_data[:, c])

            # note: if this is parallelized, this must be replaced with the list of sets to be intersected
            #  after the loop, which would entail making a list of sets of indices and intersecting with
            #  candidate_indices = set.intersect(*indices_in_range) after the parallel executions finish
            index_range = index_table[table_low:table_high, c]
            if c == 0:  # catch the first loop to initialize the indices
                candidate_indices = index_range
            else:
                candidate_indices = np.intersect1d(candidate_indices, index_range)

        if len(candidate_indices) >= num_data_points:
            data_indices = candidate_indices  # this run is the new best, so save the results
            # if the right number of points have been found, stop changing alpha
            if len(candidate_indices) != num_data_points:
                best_high_alpha = current_alpha
                current_alpha -= (best_high_alpha - best_low_alpha) / 2
        else:
            best_low_alpha = current_alpha
            current_alpha += (best_high_alpha - best_low_alpha) / 2

        num_iterations += 1

    return current_alpha, list(np.sort(data_indices))  # return data_indices as sorted list


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

        index_table_1 = generate_index_table(some_data_1)
        base_fuzzy = get_base_fuzzy(some_data_1)
        an_input_1 = np.array([5, 5])

        # easiest test with small data, all in order
        values_1 = get_alpha(an_input_1, some_data_1, index_table_1, base_fuzzy, 4, 10)
        assert(values_1[0] * 8 < 3)
        print("values_1 indices:", values_1[1])
        assert(values_1[1] == [2, 3, 4, 5])

        some_data_2 = [[1, 9, 1],
                       [2, 8, 1],
                       [3, 7, 1],
                       [4, 6, 2],
                       [6, 4, 2],
                       [7, 3, 3],
                       [8, 2, 3],
                       [9, 1, 3]]

        index_table_2 = generate_index_table(some_data_2)

        # same test as 1, but data is inherently non-sortable
        values_2 = get_alpha(an_input_1, some_data_2, index_table_2, base_fuzzy, 4, 10)
        assert(values_2[0] * 8 < 3)
        assert(values_2[1] == [2, 3, 4, 5])

        # same test as 2 but finding fewer points of output
        values_3 = get_alpha(an_input_1, some_data_2, index_table_2, base_fuzzy, 2, 10)
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
                       [5.1, 3.6, 2.6, 1.6, 2.01]]
        index_table_4 = generate_index_table(some_data_4)
        an_input_4 = np.array([4.5, 3.0, 2.0, 1.0])
        base_fuzzy_4 = get_base_fuzzy(some_data_4)

        values_4 = get_alpha(an_input_4, some_data_4, index_table_4, base_fuzzy_4, 4, 10)
        assert(values_4[0] * 1.2 < 0.3)
        assert(values_4[1] == [4, 5, 6, 7])

        an_input_5 = np.array([5.2, 3.7, 2.7, 1.7])

        # edge case: input is after the edge of the dataset
        values_5 = get_alpha(an_input_5, some_data_4, index_table_4, base_fuzzy_4, 4, 10)
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
                       [5.1, 3.6, 2.6, 1.6, 2.01]]

        index_table_6 = generate_index_table(some_data_6)

        values_6 = get_alpha(an_input_4, some_data_6, index_table_6, base_fuzzy_4, 4, 10)
        assert(values_6[0] * 1.2 < 0.5)
        print("values_6[1]", values_6[1])
        assert(values_6[1] == [1, 2, 3, 8])

        # this shows that the same data will find the same lines for the same input
        # regardless of the order of the data
        for i in range(4):
            assert(some_data_4[values_4[1][i]][i] == some_data_6[values_6[1][i]][i])

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

        index_table_7 = generate_index_table(some_data_7)
        an_input_7 = np.array([5.7, 2.6, 3.5, 1.0])  # target for this input is 2.0
        base_fuzzy_7 = get_base_fuzzy(some_data_7)

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
        values_7 = get_alpha(an_input_7, some_data_7, index_table_7, base_fuzzy_7, 4, max_iterations=10)
        print(values_7[0], "<- alpha  7  indices ->", values_7[1])
        assert(abs(values_7[0] - 0.2) < self.DELTA)
