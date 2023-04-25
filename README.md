# Project MaRz
### This repository contains the core code files of the MaRz project.
Files here are working and tested code and are used in the MaRz project.
Some aspects of the project are still in development, but everything here
is working and not testing files.

## Dependencies
* `numpy` for dataset management
* `unittest` for testing
* `sklearn` for splitting and analysis
#### For specific experiments
* `matplotlib.pyplot` and `seaborn` for charts
* `tensorflow` for heart disease comparison

## How to Use
The process of making a query with MaRz happens in three phases: Preprocessing, 
Hyper-boxing, and Querying. For full tests, entire small (~5000 lines or fewer) datasets
can be processed all together.

### Format and Preprocessing
In order to ensure compatibility, datasets should be formatted as 2D numpy arrays with rows of inputs
and columns of attributes. The last column of the dataset should be the target outputs of the data.

The preprocessing done in `dataset_preprocessing.generate_index_table` creates a sorter table of indices
of the dataset and fills it with columns where-in each column is the indices of the corresponding column
of the data set, in the order they would be in if the dataset were sorted (stable) by that column.

The second part of preprocessing is to generate the base fuzzy width for the dataset. This
is done with `dataset_preprocessing.get_base_fuzzy`, which takes a properly formatted dataset
and returns a list of value ranges for each feature of the dataset, referred to as the `base_fuzzy`.

### Hyper-boxing
At this point, an input is needed. An input is a list of features the same size as a single
entry in the dataset, but without a target on the end.  

Generating an appropriate hyperbox for the data is a small tuning process, which uses the
`get_alpha` function in `get_alpha_sorted.py`. This version uses an index table as generated
in the preprocessing. The unsorted version does not use an index table, but it less efficient.  

The `alpha` value is a number between 0 and 1 by which each number of the `base_fuzzy` is
multiplied in order to create a hyperbox around the input within the dataset which contains
the input as well as a minimum of `n` additional datapoints, as indicated by the `points` argument.
`get_alpha` also returns a container of the indices of the points within the hyper-box.

### Querying
To query MaRz with the chosen input uses the `marz_get_output.get_output` function, which
applies the fuzzy calculation to the points in the hyper-box and produces a prediction
for the input. This output is a decimal value appropriate to the targets of the dataset.

### Full Tests
The `run_dataset.py` file makes it convenient to process a full dataset and get back two lists
containing the real and predicted values returned when each line of a dataset is given as input
to the MaRz process. These output lists can be used to calculate an accuracy score, with the
entire dataset as "testing split." See this in code in the `airfoil_data.py` experiement.

In order to handle columns where every value is the same, the `base_fuzzy`
for that column is converted from 0 to 0.000001, to prevent division by 0 downstream.

## Citation
All code, unless otherwise indicated, was written by Seth Gorrin at the direction of Dr. Eric Braude.
External sources are cited where they are used. 
