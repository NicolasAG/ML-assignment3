# ML-assignment3

### Guidelines

Consider the data set available in the file `hw3pca.txt`: each row represents an instance and the columns represent features.

- split the data into 80% representing the training set and 20% to test the representation.
- Perform PCA on the data
- plot the reconstruction error as a function of the number of dimensions: both on the training set and on the test set
- plot the fraction of the variance accounted for obtained by looking at the top eigenvalues.

Explain what you see and what are the implications for choosing dimensionality of the data.


### Run script

`python a3q1.py <FLAGS>`

FLAGS:
- `--visualize` will plot the data in 1D, 2D, and 3D
- `--verbose` will print out matrix shape information and mean error for each dimension
