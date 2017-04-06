import argparse
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""

Consider the data set available in the file hw3pca.txt;
each row represents an instance and the columns represent features.

1) split the data into 80% representing the training set and 20% to test the representation.

2) Perform PCA on the data

3) plot the reconstruction error as a function of the number of dimensions,
3.1) both on the training set
3.2) and on the test set,

4) plot the fraction of the variance accounted for obtained by looking at the top eigenvalues.

Explain what you see and what are the implications for choosing dimensionality of the data.

"""


def parse_args():
    parser = argparse.ArgumentParser(description='COMP 652 - Machine Learning - Assignment 3')
    parser.add_argument('--visualize', action="store_true", help='produce 1D, 2D, 3D visualization plots of the data')
    parser.add_argument('--verbose', action="store_true", help='print details for each PCA reduction')
    args = parser.parse_args()
    return args

def visualize1D(data):
    pca = PCA(n_components=1)
    pca.fit(data)
    data_pca = pca.transform(data)
    plt.title("Data visualisation in 1D")
    plt.plot(data_pca, np.zeros(data_pca.shape, dtype=int), 'bo')
    plt.show()


def visualize2D(data):
    pca = PCA(n_components=2)
    pca.fit(data)
    data_pca = pca.transform(data)
    plt.title("Data visualisation in 2D")
    plt.plot(data_pca[:, 0], data_pca[:, 1], 'bo')
    plt.show()


def visualize3D(data):
    pca = PCA(n_components=3)
    pca.fit(data)
    data_pca = pca.transform(data)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_title("Data visualisation in 3D")
    ax.plot(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2], 'bo')
    plt.show()


def main(args):
    print "\nLoading data..."
    with open('hw3pca.txt', 'r') as f:
        lines = f.readlines()
        m = len(lines)  # number of instances
        n = len(lines[0].split())  # number of features
        x = np.zeros((m, n), dtype=float)
        for i, line in enumerate(lines):
            features = line.split()
            assert len(features) == n
            x[i] = features
    if args.verbose: print "x:", x.shape

    # 1) split data into 80% train 20% test set.
    train_size = int(m * 0.8)
    x_train = np.array(x[:train_size])
    x_test = np.array(x[train_size:])
    if args.verbose: print "x_train:", x_train.shape
    if args.verbose: print "x_test:", x_test.shape
    print "done."

    if args.visualize:
        visualize1D(x)
        visualize2D(x)
        visualize3D(x)

    plt_x_axis = []
    plt_y_axis_train = []
    plt_y_axis_test = []
    print "\nPerforming PCA with k from 1 to %d..." % min(x_train.shape[0], n)
    # perform PCA on x_train
    for k in range(1, min(x_train.shape[0], n)+1):  # k is the number of dimensions to keep
        if args.verbose: print "\nPCA with %d components:" % k
        pca = PCA(n_components=k)
        pca.fit(x_train)
        # Transform training & test set with learnt PCA mapping
        x_train_pca = pca.transform(x_train)
        x_test_pca = pca.transform(x_test)
        if args.verbose: print "x_train_pca:", x_train_pca.shape
        if args.verbose: print "x_test_pca:", x_test_pca.shape
        # Reconstruct training & test set from the PCA data
        x_train_projected = pca.inverse_transform(x_train_pca)
        x_test_projected = pca.inverse_transform(x_test_pca)
        if args.verbose: print "x_train_projected:", x_train_projected.shape
        if args.verbose: print "x_test_projected:", x_test_projected.shape
        # Measure mean reconstruction error for each feature = mean over all instance i (x[i] - x'[i])**2
        train_reconstruction_error = np.mean((x_train - x_train_projected) ** 2, axis=0)
        test_reconstruction_error = np.mean((x_test - x_test_projected) ** 2, axis=0)
        if args.verbose: print "train reconstruction error:", train_reconstruction_error.shape
        if args.verbose: print "test reconstruction error:", test_reconstruction_error.shape
        # Overall mean reconstruction error:
        train_reconstruction_mean_error = np.mean(train_reconstruction_error)
        test_reconstruction_mean_error = np.mean(test_reconstruction_error)
        if args.verbose: print "mean train error:", train_reconstruction_mean_error
        if args.verbose: print "mean test error:", test_reconstruction_mean_error
        # Add to pyplot arrays:
        plt_x_axis.append(k)
        plt_y_axis_train.append(train_reconstruction_mean_error)
        plt_y_axis_test.append(test_reconstruction_mean_error)
    print "done."

    # Plot reconstruction error for training and test set
    plt.title("Mean reconstruction error")
    line1, = plt.plot(plt_x_axis, plt_y_axis_train, 'bo-')
    line2, = plt.plot(plt_x_axis, plt_y_axis_test, 'go-')
    plt.legend([line1, line2], ['train', 'test'])
    plt.xlabel('number of dimensions')
    plt.ylabel('average reconstruction error')
    plt.xscale('log')
    plt.show()

    # Get the explained variance
    explained_variance = pca.explained_variance_ratio_
    if args.verbose: print "\nExplained variance (normalized):", explained_variance.shape

    # Plot the explained variance
    plt.title("Explained variance")
    plt.plot(plt_x_axis, explained_variance, 'ro-')
    plt.xlabel('number of dimensions')
    plt.ylabel('eigenvalues')
    plt.xscale('log')
    plt.show()

if __name__ == '__main__':
    args = parse_args()
    print args
    main(args)
