# Project MaRz
### This repository contains some core code files of the MaRz project.
Files here are working and tested code which is used in the MaRz project and
ready to show the world. They may still be in development, but they are working
versions and not testing files.

## Dependencies
* `numpy` (for dataset management)
* `unittest` (for testing)

## Publicity
This is currently a private repository. Proper project documentation and crediting
will need to be added if it is ever made public.


### In-progress notes
`get_alpha_sorted` is working, but running full-scale tests requires some extra steps
in order to run. For each removed (testing) line of the dataset, the index table needs
to not have that line. If the line is simply deleted from the dataset, then that index
number must be removed from the table and then all indices after it must be shifted back
to remove the gap.

My current best alternative is to change the testing line to out-of-range values, and then
move each instance of that line to the end of the index table. This should work, but I am
looking for an efficient way to accomplish it, since the current plan involves a lot of
operations on the index table.