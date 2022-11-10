# Project MaRz
### This repository contains some core code files of the MaRz project.
Files here are working and tested code which is used in the MaRz project and
ready to show the world. They may still be in development, but they are working
versions and not testing files.

## Dependencies
* `numpy` (for dataset management)
* `unittest` (for testing)

## Format and Preprocessing
In order to ensure compatibility, datasets should be formatted as 2D numpy arrays with rows of inputs
and columns of attributes. The last column of the dataset should be the target outputs of the data.

The preprocessing done in `dataset_preprocessing.generate_index_table` creates a sorter table of indices
of the dataset and fills it with columns where-in each column is the indices of the corresponding column
of the data set, in the order they would be in if the dataset were sorted (stable) by that column. This
step also removes any columns from the dataset for which every value in the column is the same. All
steps of the MaRz process assume that every column of data contains at least two distinct values.

## Publicity
This is currently a private repository. Proper project documentation and crediting
will need to be added if it is ever made public.
