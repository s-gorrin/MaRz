"""
Build a query-able dataset in real time.
"""

import numpy as np
import time


class RealtimeDataCollector:

    input_dataset = None  # to store a dataset for experiments

    def __init__(self, num_attributes):
        self.data_width = num_attributes

        # setup empty lists
        self.data_lists = []
        self.sort_lists = []
        self.setup_lists()

    def setup_lists(self):
        """
        INTENT: Set up empty lists for data and sorter.
        """
        # clear lists just in case
        self.data_lists = []
        self.sort_lists = []

        # fill empty lists with empty lists
        for i in range(self.data_width):
            self.sort_lists.append([])
            self.data_lists.append([])

    def full_dataset(self, ds):
        self.input_dataset = ds

    # make a thread object, start it with thread.run (or thread.start?)
    def realtime_data_input(self):
        """
        INTENT: Add data to the data_ and sort_lists in simulated "real time."

        PRE 1: self.input_dataset contains a dataset
        """
        # prime class variables for the dataset
        self.data_width = self.input_dataset.shape[1]
        self.setup_lists()

        for i in range(self.input_dataset.shape[0]):  # rows
            self.add_datum(self.input_dataset[i, :])
            print('.', end='')
            time.sleep(0.24)  # might be replacing this with threading.Event.wait()

    def add_datum(self, a_datum):
        """
        INTENT: Add a a_datum to the data_lists and sort_lists.

        PRE 1: a_datum is a list or array of len() == self.data_width.

        POST 1: The a_datum is appended to the data_lists.
        POST 2: The a_datum's index is sorted into sort_lists.
        """
        datum_id = len(self.data_lists[0])

        if len(self.data_lists[0]) > 0:
            # the item is sorted into the sort_lists and then added to the data_lists
            for i in range(self.data_width):
                index = np.searchsorted(self.data_lists[i], a_datum[i], sorter=self.sort_lists[i])
                # edge case: new index is at the end
                if index == len(self.sort_lists[i]):
                    self.sort_lists[i].append(datum_id)
                else:
                    self.sort_lists[i].insert(index, datum_id)
                self.data_lists[i].append(a_datum[i])
        # edge case: adding the first item to the lists
        else:
            for i in range(self.data_width):
                self.data_lists[i].append(a_datum[i])
                self.sort_lists[i].append(0)

    def get_data(self):
        """
        INTENT: Get the data as a numpy array.
        """
        return np.array(self.data_lists).transpose()

    def get_sorter(self):
        """
        INTENT: Get the sorter as a numpy array.
        """
        return np.array(self.sort_lists).transpose()


if __name__ == '__main__':
    arr = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [3, 7], [9, 2], [1, 1], [11, 11]]
    stream_collector = RealtimeDataCollector(2)
    for row in arr:
        stream_collector.add_datum(row)

    print("data:", stream_collector.data_lists)
    print("data as array:\n", stream_collector.get_data())
    print()
    print("sorter:", stream_collector.sort_lists)
    print("sorter as array:\n", stream_collector.get_sorter())
