from dataset_preprocessing import *
from get_alpha_sorted import get_alpha
from new_get_output import get_output

"""
This allows full datasets to be run consistently for testing purposes.
"""


def preprocessing(some_data):
    """
    INTENT: do the preprocessing steps for running a dataset
    RETURN: the index_table and base_fuzzy
    """
    updated_data, index_table = generate_index_table(some_data)
    base_fuzzy = get_base_fuzzy(updated_data)

    return updated_data, index_table, base_fuzzy


def run_dataset(some_data, index_table, base_fuzzy):
    """
    INTENT: the procedural work of running a full set of tests on a dataset and printing results

    This uses the get_alpha sorted version
    """
    # print some_data info
    print(f"data shape: {some_data.shape}")
    print(f"base fuzzy: {base_fuzzy}")

    length = some_data.shape[0]

    points = 2
    close_threshold = 0.5
    close = 0
    close_lines = []
    print_lines = False

    print(f'points sought: {points}, threshold for "close result": {close_threshold}')

    for i in range(length):
        trimmed_data, trimmed_table, test = extract_test_line(some_data, index_table, i)

        alpha, indices = get_alpha(test[0], trimmed_data, trimmed_table, base_fuzzy, points, max_iterations=10)
        output = get_output(test[0], trimmed_data, base_fuzzy * alpha, indices)

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

    print(f"{close} close results of {length}, or {(close / length) * 100:.2f}%")
