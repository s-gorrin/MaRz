"""
Query a dataset in real time.
"""
from sklearn.model_selection import train_test_split
import time
import threading

import dataset_preprocessing
from RealtimeDataCollector import RealtimeDataCollector
from get_alpha_sorted import get_alpha
from marz_get_output import get_output
from DatasetSelection import DatasetSelection


def split_dataset(a_dataset, test_size=0.2):
    """
    INTENT: Do a train/test split on a_dataset in order to perform real-time experiment.

    PRECONDITION 1: a_dataset is formatted for MaRz.
    PRE 2: test_size is the portion of the data to put in the test set, 0 < test_size < 1

    POSTCONDITION 1: The dataset is spilt and the two parts returned.
    """
    train_set, test_set = train_test_split(a_dataset, test_size=test_size,
                                           random_state=0, shuffle=False)
    return train_set, test_set


def setup_realtime_collector(train_set):
    """
    INTENT: Create an instance of a RealtimeDataCollector with the training set.

    PRECONDITION 1: train_set is a dataset appropriately formatted for MaRz operations.

    POSTCONDITION 1: a RealtimeDataCollector is instantiated and supplied with
        the train_set, and then returned.
    """
    global collector
    collector = RealtimeDataCollector(len(train_set[0, :]))
    collector.full_dataset(train_set)

    return collector


def query_input(an_input, the_dataset, the_sorter):
    """
    INTENT: Query a current iteration of the real-time dataset with a single input.

    PRECONDITION 1: an_input could be a line of the_dataset.
    PRE 2: the_sorter is an index table for the_dataset.

    POSTCONDITION 1: the_database is queried with an_input and the output is returned.
    """
    base_fuzzy = dataset_preprocessing.get_base_fuzzy(the_dataset)

    alpha, indices = get_alpha(an_input, the_dataset, the_sorter, base_fuzzy, 1, 10)
    return get_output(an_input, the_dataset, base_fuzzy * alpha, indices)


# loop over the test split and query the collector, printing results as they come
def send_test_set(test_set):
    """
    INTENT: In a thread, query the collected dataset with each item of the test_set.

    PRECONDITION 1: test_set is a portion of a complete dataset, of which the other
        portion is the training set, which is populating the realtime collector.

    POSTCONDITION 1: The collector is accessed to retrieve the current version of
        the sorter and dataset.
    POST 2: Each line of the test_set is given as input to the current version of the dataset.
    POST 3: Each time a query is made, the result is printed, with a note indicating if it is correct.
    POST 4: The thread is paused between queries.
    """
    for line in test_set:
        target = line[-1]

        # ---- POST 1
        global collector
        sorter = collector.get_sorter()
        data = collector.get_data()
        # The only way they should not be equal is if the data has had an extra row
        #   added that is not in the sorter yet. Thus, get the sorter first, and
        #   then the data, and if they're not the same, trim the extra row off the data
        if not (data.shape == sorter.shape):
            # this happens infrequently, but it can happen and is necessary
            data = data[:-1]

        # ---- POST 2
        output = query_input(line[:-1], data, sorter)
        success = abs(target - output) < 0.5

        # ---- POST 3
        print(f"\tfor target {int(target)}, got {output:.1f}{', which is correct!' if success else ''}")

        # ---- POST 4
        # one query per second seems like a reasonable pace
        #   and leaves plenty of time for the collector
        time.sleep(1)


if __name__ == '__main__':
    """
    INTENT: Run the realtime experiment on the digits dataset.
    
    POSTCONDITION 1: The dataset is retrieved a split into a training set and a testing set.
    POST 2: The collector is set up with the training set and a thread is initialized.
    POST 3: The threads are started to run the experiment and results are printed to the console.
    """

    # ---- POST 1
    training_set, testing_set = split_dataset(DatasetSelection('digits').dataset)

    # ---- POST 2
    collector = setup_realtime_collector(training_set)
    # set up a thread for filling up the training dataset
    collection_thread = threading.Thread(target=collector.realtime_data_input)

    print("Starting Experiment...")
    print("Each '.' represents one additional datapoint added to the training set.\n"
          "Every time a query is made, the result is printed and marked as correct if it is.")

    # ---- POST 3
    # start the collector
    collection_thread.start()
    # the query process can use the main thread
    send_test_set(testing_set)
