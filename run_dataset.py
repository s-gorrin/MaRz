from dataset_preprocessing import *
from get_alpha_sorted import get_alpha
from marz_get_output import get_output

"""
This allows full datasets to be run consistently for testing purposes.
"""


def extract_test_line(some_data, index_table, test_row_number):
    """
    INTENT: create a "test split" for the some_data by extracting the test_row_number from the data and index table

    PRE 1: some_data is a some_data and index_table is a sorter table for that some_data
    PRE 2: some_data and index_table are the same shape as defined by numpy
    PRE 3: test_row_number is an index within some_data and index_table

    POST 1: the line indicated by test_row_number is removed from some_data
    POST 2: test_row_number is removed from the index_table and later rows are shifted up to fill in the gap

    RETURN: the reduced some_data, the reduced sorter, the test line
    """
    row = some_data[test_row_number]
    an_input = (row[:-1], row[-1])  # test row as a tuple: input, target
    trimmed_data = np.delete(some_data, test_row_number, axis=0)  # remove the test row

    # remove the test_row_number, a critical step
    tp = index_table.T
    tp = tp[tp != test_row_number].reshape(trimmed_data.shape[1], trimmed_data.shape[0]).T
    trimmed_table = np.subtract(tp, 1, where=(tp > test_row_number), out=tp.copy())

    return trimmed_data, trimmed_table, an_input


def perc(x, total):
    # make getting percentages easier since it's happening a few times
    return (x / total) * 100


def preprocessing(some_data):
    """
    INTENT: do the preprocessing steps for running a dataset
    RETURN: the index_table and base_fuzzy
    """
    index_table = generate_index_table(some_data)
    base_fuzzy = get_base_fuzzy(some_data)

    return index_table, base_fuzzy


def run_dataset(some_data, index_table, base_fuzzy, points=2, close_threshold=0.1, start=0, step=1, verbose=False):
    """
    INTENT: the procedural work of running a full set of tests on a dataset and printing results

    The arguments with default values have been added to give more control to the calling code,
        which is particularly useful for jupyter notebooks.

    start and step are used in the range() for which lines of the dataset to run.

    This uses the get_alpha sorted version
    """
    # print some_data info
    print(f"data shape: {some_data.shape}")
    # print(f"base fuzzy: {base_fuzzy}")

    length = some_data.shape[0]

    close = 0
    close_lines = []
    print_lines = verbose
    lines_run = 0
    points_count = [0, 0, 0]
    # gathering both of these together in case of partial runs
    targets = []
    outputs = []

    for i in range(start, length, step):
        trimmed_data, trimmed_table, test = extract_test_line(some_data, index_table, i)

        alpha, indices = get_alpha(test[0], trimmed_data, trimmed_table, base_fuzzy, points, max_iterations=10)
        output = get_output(test[0], trimmed_data, base_fuzzy * alpha, indices)

        # increment appropriate counter to track the number of times the requested number of points was found
        if len(indices) == points:
            points_count[0] += 1
        elif len(indices) == points + 1:
            points_count[1] += 1
        else:
            points_count[2] += 1

        difference = abs(output - test[1])
        if difference < close_threshold:
            end = f"\t <- within {close_threshold}"
            close += 1
            close_lines.append(i)
        else:
            end = ""

        # print results
        if print_lines:
            print(f"{i:4}) alpha: {alpha:.3f}, points: {len(indices):2}\t"
                  f"output: {output:.3f}\ttarget: {test[1]:.3f}\tdiff: {difference:.3f}{end}")

        targets.append(test[1])  # add target, output to the loss vectors
        outputs.append(output)   # to compute accuracy (R-Square, Mean Square, etc)
        lines_run += 1

    print(f'threshold for "close result": {close_threshold}')
    print(f"{close} close results of {lines_run} lines, or {perc(close, lines_run):.2f}%")
    print(f"points requested: {points}\nnumber with {points} points found: "
          f"{points_count[0]} or {perc(points_count[0], lines_run):.2f}%\n"
          f"number with {points + 1} points found: {points_count[1]} or {perc(points_count[1], lines_run):.2f}%\n"
          f"number with more points found: {points_count[2]} or {perc(points_count[2], lines_run):.2f}%")

    return targets, outputs
