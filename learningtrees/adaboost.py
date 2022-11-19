from time import time

import matplotlib.pyplot as plt
import numpy as np

from learningtrees.decisiontree import DecisionTree


class AdaBoost:
    """Adaboost classifier.

    Uses a decision tree as the weak classifier.

    Attributes:
        X (np.ndarray): Training data.
        y (np.ndarray): Training labels.
        n_trees (int): Number of trees to use.
        max_depth (int): Max depth of all trees.
    
    Methods:
        fit: Fit model.
        predict: Predict using the model.
        score: Score the model.
    """

    def __init__(self, X:np.ndarray, y:np.ndarray, num_trees:int=100, max_depth=4, **kwargs):
        """Constructor

        Args:
            X (np.ndarray): Training data.
            y (np.ndarray): Training labels.
            num_trees (int, optional): Number of trees to use. Defaults to 100.
            max_depth (int, optional): Max depth of all trees. Defaults to 4.
        
        Keyword arguments:
            min_samples_split (int, optional): The minimum number of samples to split a node. Defaults to 4.
            min_impurity (float, optional): The minimum impurity to split a node. Defaults to 0.1.
            impurity_measure (str): The impurity measure to use. Defaults to "gini".
        """

        self.X = X
        self.y = y
        self.n_trees = num_trees
        self.max_depth = max_depth

        # Keep track of the weak classifiers
        self.weak_classifiers = {"model": {}, "training_error": np.zeros((self.n_trees,)), "oob_error": np.zeros((self.n_trees,)), "alphas": {}}

        self.impurity_measure = kwargs.get("impurity_measure", "gini")
        self.min_samples_split = kwargs.get("min_samples_split", 5)
        self.min_impurity = kwargs.get("min_impurity", 0.1)

    def fit(self, logging=False, plot_training=False) -> None:
        """Fit model."""

        # Start by converting 0 labels to -1, retaining 1 labels
        self.y = self.convert_labels(self.y, backwards=True)

        # Initialize weights, uniformly at first
        w_i = np.ones(self.y.shape[0]) / self.y.shape[0]

        for i in range(self.n_trees):
            
            # Generate a subset of the training data, with replacement. After one iteration, the subset will be weighted.
            # We can also use the data points which was not sampled to calculate the Out-of-bag error.
            X, y, out_of_bag_ind = self._weighted_sample(self.X, self.y, w_i)
            
            # As the decision tree does not accept negative labels, convert them to 0.
            y = self.convert_labels(y)

            # Fit the decision tree on the sampled data
            weak_classifier = DecisionTree(
                X,
                y,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_impurity=self.min_impurity,
                impurity_measure=self.impurity_measure
            )

            weak_classifier.fit()

            # Predict the training data labels
            predictions = weak_classifier.predict(self.X)

            # Calculate out of bag error
            oob_error = np.sum(predictions[out_of_bag_ind] != self.y[out_of_bag_ind]) / out_of_bag_ind.shape[0]

            # Convert predictions back to -1 instead of 0
            predictions = self.convert_labels(predictions, backwards=True)

            # Calculate the missclassifications, for the case where a data point was not sampled, set the value to 0, and if correctly classified, set the value to 0.
            missclassified = np.where(
                predictions != 0,                       # Condition
                np.where(predictions != self.y, 1, 0),  # Where condition is true
                0                                       # Where condition is false
            )

            # Calculate the error as the sum of the weights of the missclassified data points
            err = np.sum(w_i * missclassified)

            if logging:
                print(f"err: {err:.3f}, oob_error: {oob_error:.3f}")

            # Calculate alpha
            alpha = 0.5 * np.log((1 - err) / err)

            # Update the weights
            w_i = w_i * np.exp(-alpha * predictions * self.y)
            w_i = w_i / np.sum(w_i)

            # Save this run
            self.weak_classifiers["model"][i] = weak_classifier
            self.weak_classifiers["training_error"][i] = err
            self.weak_classifiers["oob_error"][i] = oob_error
            self.weak_classifiers["alphas"][i] = alpha

            if plot_training:
                self._plot_training_error(i)
        
        if plot_training:
            plt.show()

    def _plot_training_error(self, step:int):
        """Plot the training and Out-of-Box error.

        Args:
            step (int): Current step.
        """

        terr = self.weak_classifiers["training_error"][:step]
        oob = self.weak_classifiers["oob_error"][:step]
        steps = np.arange(1, step+1)

        plt.clf()
        plt.xlim(0, self.n_trees)
        plt.ylim(0, 1)
        plt.axhline(y=0.5, color="r", linestyle="--")
        plt.plot(steps, terr, color="r", linestyle="-", label="Training error")
        plt.plot(steps, oob, color="b", linestyle="-", label="Out-of-bag error")
        plt.title(f"Training and Out-of-Bag error at step {step+1}")
        plt.xlabel("Step")
        plt.ylabel("Error")
        plt.legend()
        plt.pause(0.01)

    @staticmethod
    def predict_n(X:np.ndarray, classifiers:list, alphas:list) -> np.ndarray:
        """Prediction using given classifiers
        
        Args:
            X (np.array): Data to predict.
            classifiers (list): list containing classifiers with keys "alphas" and "model".
            alphas (list): list of alphas

        Returns:
            (np.array): Predictions.
        """

        predictions = np.zeros(X.shape[0])

        for i, clf in enumerate(classifiers):
            predictions += alphas[i] * AdaBoost.convert_labels(clf.predict(X), backwards=True)
        
        return np.sign(predictions)

    def predict(self, X:np.ndarray) -> np.ndarray:
        """Predict using the model
        
        Args:
            X (np.array): Data to predict.
        
        Returns:
            (np.array): Predictions.
        """

        return self.predict_n(X, list(self.weak_classifiers["model"].values()), list(self.weak_classifiers["alphas"].values()))

    def score(self, X:np.ndarray, y:np.ndarray, predictor=None) -> float:
        """Score the model
        
        Args:
            X (np.ndarray): Data to score.
            y (np.ndarray): Labels to compare with.

        Returns:
            float: Score.
        """

        if predictor is None:
            predictions = self.predict(X)
        else:
            predictions = predictor(X)

        return np.sum(predictions == self.convert_labels(y, backwards=True)) / y.shape[0]

    @staticmethod
    def _weighted_sample(X:np.ndarray, y:np.ndarray, weights:np.ndarray) -> tuple:
        """Return the weighted sampled dataset X with replacement.
        Args:
            X (np.array): Data set, shape (n x m)
            y (np.array): Labels, shape (n,)
            weights (np.array): weights, shape (n,)
        """
        samples = np.arange(X.shape[0])
        sampled_indices = np.random.choice(samples, size=X.shape[0], replace=True, p=weights)

        out_of_bag_index = np.setdiff1d(samples, sampled_indices)

        return X[sampled_indices], y[sampled_indices], out_of_bag_index

    @staticmethod
    def convert_labels(y:np.ndarray, backwards:bool=False) -> np.ndarray:
        """Convert negative labels to 0 in forwards mode and 0 labels to -1 in backwards mode.
        
        Args:
            y (np.array): label array
            backwards (bool, optional): True means convert 0 to -1. Defaults to False.

        Returns:
            y (np.array): label array with fixed labels
        """

        if backwards:
            y[y==0] = -1
            return y

        y[y<0] = 0
        return y