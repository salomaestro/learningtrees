from time import time

import numpy as np

from learningtrees.baggingtree import BaggingTree
from learningtrees.decisiontree import DecisionTree
from learningtrees.random_subspace_method import RandomSubspaceMethod


class RandomForest(BaggingTree):
    """Random forest decision tree implementation which extends BaggingTree class.
    
    Attributes:
        ...
    Methods:
        fit(logging=False): ...
        predict(X): ...
        score(X, y): ...
    """

    def __init__(self, X, y, n_subsets=1, **kwargs):
        """Initialize RandomForest object.

        Args:
            X (np.array): data set
            y (np.array): labels
            n_subsets (int, optional): how many different trees will be sampled for. Defaults to 1.

        
        Keyword Args:
            max_depth (int, optional): The maximum depth of the tree. Defaults to 10.
            min_samples_split (int, optional): The minimum number of samples to split a node. Defaults to 4.
            min_impurity (float, optional): The minimum impurity to split a node. Defaults to 0.1.
            impurity_measure (str, optional): The impurity measure to use. Defaults to "gini".
            n_samples (int): number of samples to draw (with replacement) from X along axis 0. Defaults to length - 1 of X along axis 0.
            n_features (int): number of features to draw (with replacement) from X along axis 1. Default to length - 1 of X along axis 1.
        """

        super().__init__(X, y, **kwargs)

        n_samples = min(kwargs.get("n_samples", X.shape[0] - 1), X.shape[0] - 1)
        n_features = min(kwargs.get("n_features", X.shape[1] - 1), X.shape[1] - 1)

        randomsubspace = RandomSubspaceMethod(X, n_subsets=n_subsets, n_samples=n_samples, n_features=n_features)

        self.sample_mat = randomsubspace.sample_matrix
        self.feature_mat = randomsubspace.feature_matrix

        self._n_trees = n_subsets

        self.params = kwargs

    def fit(self, logging=False) -> dict:
        """Fit the RandomForest object.
        
        Returns:
            trees: dictionary of trees with their respective samples indices and feature indices.
        """

        self.trees = {}

        for i, (samples_idx, feature_idx) in enumerate(zip(self.sample_mat.T, self.feature_mat.T)):

            X_train = self.X[samples_idx][:, feature_idx]
            y_train = self.y[samples_idx]

            tree = DecisionTree(
                X_train,
                y_train,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_impurity=self.min_impurity,
                impurity_measuer=self.impurity_measure
            )

            fit_start = time()

            # Fit the tree to the data.
            tree.fit()

            if logging:
                print(f"============= Done training tree {i}, took {(time() - fit_start)*1e3:.0f} ms. Reached depth {tree.deepest_node} =============")

            self.trees[i] = {"tree": tree, "samples": samples_idx, "features": feature_idx}

        return self.trees

    def predict_probas(self, X:np.ndarray) -> np.ndarray:
        """Predict the probability of each class for each input entry for a random forest tree."""

        prediction_mat = np.zeros((X.shape[0], self._n_trees)).astype(np.float64)

        for i, tree_dict in enumerate(self.trees.values()):
            tree = tree_dict["tree"]
            features = tree_dict["features"]

            prediction_mat[:, i] = tree.predict_probas(
                X[:, features] # Snip away the features not used in fitting the tree.
            )[:, 1]
        
        # probas = np.sum(prediction_mat, axis=1) / self._n_trees
        probas = np.median(prediction_mat, axis=1)

        return probas

    def predict(self, X:np.ndarray) -> np.ndarray:
        """Predict class of each input entry for a random forest tree."""

        prediction_mat = np.zeros((X.shape[0], self._n_trees)).astype(np.int64)

        for i, tree_dict in enumerate(self.trees.values()):
            tree = tree_dict["tree"]
            features = tree_dict["features"]

            try:
                prediction_mat[:, i] = tree.predict(
                    X[:, features] # Snip away the features not used in fitting the tree.
                )
            except Exception as e:
                print(e)
                print(X.shape)
                print(prediction_mat.shape)
                print(features.shape)
                exit(1)
        
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