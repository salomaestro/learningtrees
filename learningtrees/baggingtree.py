from time import time

import numpy as np

from util.decisiontrees.bootstrap import BootstrapData
from util.decisiontrees.decisiontree import DecisionTree


class BaggingTree:
    """Classify using a bagging -bootstrap aggregate- decision tree.
    """

    def __init__(self, X, y, **kwargs):
        """Initialize BaggingTree object.
        
        Args:
            X (np.array): data set
            y (np.array): labels

        Keyword Args:
            max_depth (int, optional): The maximum depth of the tree. Defaults to 10.
            min_samples_split (int, optional): The minimum number of samples to split a node. Defaults to 4.
            min_impurity (float, optional): The minimum impurity to split a node. Defaults to 0.1.
            impurity_measure (str): The impurity measure to use. Defaults to "gini".
            max_n_bootstraps (int, optional): The number of bootstraps to use. Defaults to 30.
        """

        self.X = X
        self.y = y

        # Kwargs
        self.max_depth = kwargs.get('max_depth', 10)
        self.min_samples_split = kwargs.get('min_samples_split', 2)
        self.min_impurity = kwargs.get('min_impurity', 0.01)
        self.impurity_measure = kwargs.get('impurity_measure', 'gini')
        self.max_n_bootstraps = kwargs.get("max_n_bootstraps", 30)
        self.n_samples = kwargs.get("n_samples", self.X.shape[0])

        self.bootstrap = BootstrapData(self.X, max_bootstraps=self.max_n_bootstraps, n_samples=self.n_samples)
        # self.sample_mat, self.OOB_mat = self.bootstrap.sample_matrix, self.bootstrap.OOB_test_matrix
        self.sample_mat = self.bootstrap.sample_matrix

        self._n_trees = self.max_n_bootstraps


    def fit(self, logging=False) -> dict:
        """Fit the BaggingTree object.

        Returns:
            trees: dictionary of trees with their respective samples indices and OOB test matrix
        """

        self.trees = {}
        
        for i, samples_idx in enumerate(self.sample_mat.T):

            X_train = self.X[samples_idx]
            y_train = self.y[samples_idx]

            tree = DecisionTree(
                X_train,
                y_train,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_impurity=self.min_impurity,
                impurity_measure=self.impurity_measure
            )

            fit_start = time()

            # Fit the tree to the data.
            tree.fit()

            if logging:
                print(f"============= Done training tree {i}, took {(time() - fit_start)*1e3:.0f} ms. Reached depth {tree.deepest_node} =============")
            
            # self.trees[i] = {"tree": tree, "samples": samples_idx, "OOB": OOB_idx}
            self.trees[i] = {"tree": tree, "samples": samples_idx}

        return self.trees

    def predict(self, X:np.ndarray) -> np.ndarray:
        """Predict class of each input entry for a bagging tree."""

        prediction_mat = np.zeros((X.shape[0], self._n_trees)).astype(np.int64)

        for i, tree in enumerate(self.trees.values()):
            tree = tree["tree"]
            prediction_mat[:, i] = tree.predict(X)

        predictions = self.get_prediction(prediction_mat)

        return predictions

    def score(self, X:np.ndarray, y:np.ndarray) -> float:
        """Score the algorithm based on it's predictions compared to the actual labels y.
        
        Args:
            X (np.ndarray): Data set
            y (np.ndarray): labels
        
        Returns:
            score (float): Accuracy of model.
        """

        predictions = self.predict(X)

        return np.sum(predictions == y) / len(y)

    @staticmethod
    def get_prediction(prediction_matrix):
        """From a matrix where multiple predictions for same sample is
        located row-wise, predict as most recurring prediction in row.
        
        Args:
            prediction_matrix (np.array): matrix of predictions (n x m),
                where n is samples, and m is number of different predictions for
                sample.
        
        Returns:
            predictions (np.ndarray): prediction for each sample, shape: (n,)
        """

        return np.array(list(map(
            lambda sample: np.argmax(np.bincount(sample)),
            prediction_matrix
        )))

if __name__ == "__main__":

    # Generate some multidim test data to ensure bagging tree is working properly..
    n = 500
    cls1 = np.random.normal(loc=(0, 0, 0, 0), scale=0.5, size=(n, 4))
    cls2 = np.random.normal(loc=(0, 1, 1, 0), scale=0.5, size=(n, 4))
    cls1_y = np.zeros((cls1.shape[0]))
    cls2_y = cls1_y.copy() + 1

    X = np.vstack((cls1, cls2))
    y = np.hstack((cls1_y, cls2_y)).astype(np.int64)

    ind = np.arange(y.shape[0])

    np.random.shuffle(ind)

    test_y = y[ind[700:]]
    test_X = X[ind[700:]]

    y = y[ind[:700]]
    X = X[ind[:700]]

    bagtree = BaggingTree(X, y, max_n_bootstraps=20)

    trees = bagtree.fit()

    best_individual_score = 0

    for i, tree in enumerate(trees.values()):
        tree = tree["tree"]
        score = tree.score(test_X, test_y)
        print(f"tree {i}: score: {score:.4f}")

        if score > best_individual_score:
            best_individual_score = score

    bagscore = bagtree.score(test_X, test_y)

    print(f"bagging score: {bagscore}")

    assert bagscore > best_individual_score # Should not necessarily be true, but i want to see when it is not.
