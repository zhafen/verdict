# verdict
A class for flexible manipulation of nested data.

Acts like a dictionary with additional features, including...

* Element-wise addition, subtraction, multiplication, and addition.
* Easy export to and import from both hdf5 and json files.
* Extensive support for saving and loading jagged arrays and sparse arrays.
* Conversion to numpy arrays and pandas DataFrames.
* Easy access to contained objects' attributes.
* Nested dictionaries can be transposed, i.e. the nesting order can be changed.
* keymin and keymax, which find the extrema values in a dictionary and their keys.
* Easy subdivision into multiple dictionaries.

To install:
`pip install verdict`
