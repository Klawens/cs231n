import numpy as np
from utils.data_utils import load_CIFAR10


class k_nearest_neighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """ for kNN this is just memorizing the training data"""
        self.Xtr = X
        self.ytr = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        num_loops: Determines which implementation to use to compute distances
        between training points and testing points.
        """

        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d  for num_loops' % num_loops)
        return self.predict_labels(dists, k=k)

    """ X is testing data, self.Xtr is training data """

    def compute_distances_two_loops(self, X):
        num_test = X.shape[0]
        num_train = self.Xtr.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                dists[i, j] = np.sqrt(np.sum(np.square(self.Xtr[j] - X[i])))
                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_one_loop(self, X):
        num_test = X.shape[0]
        num_train = self.Xtr.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            dists[i, :] = np.sqrt(np.sum(np.square(self.Xtr - X[i]), axis=1))
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_no_loops(self, X):
        num_test = X.shape[0]
        num_train = self.Xtr.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops,  and store the result in     #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy.                #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using marix multiplication     #
        #       and two broadcast sums.                                         #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        M = np.dot(X, self.Xtr.T)
        te = np.square(X).sum(axis=1)
        tr = np.square(self.Xtr).sum(axis=1)
        dists = np.sqrt(-2 * M + tr + np.matrix(te).T)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            """ A list of length k storing the labels of the k nearest neighbors to
                the ith test point.
            """
            closest_y = []
            #######################################################################
            # TODO:                                                               #
            # Use the distance matrix to find the k nearest neighbors of the ith  #
            # testing point, and use self.ytr to find teh labels of these         #
            # neighbors. Store these labels in closest_y.                         #
            #                                                                     #
            # HINT: Look up the function numpy.argsort.                           #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            k_nearest = np.argsort(dists[i, :])
            closest_y = self.ytr[k_nearest[:k]]
            # #####END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #######################################################################
            # TODO:                                                               #
            # Now that you have found the labels of the k nearest neighbors, you  #
            # need to find the most common label in the list closest_y of labels. #
            # Store this label in y_pred[i]. break ties by choosing the smaller   #
            # label.                                                              #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            y_pred[i] = np.argmax(np.bincount(closest_y))
            # #####END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return y_pred
