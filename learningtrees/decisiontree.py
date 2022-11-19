"""This file contains the code for a decision tree implementation."""

from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class DecisionTree:
    def __init__(self, X:np.ndarray, y:np.ndarray, **kwargs):
        """Initialize the decision tree.
        
        Args:
            X (np.ndarray): The data to train the tree on.
            y (np.ndarray): The labels to train the tree on.
        
        Keyword Args:
            max_depth (int, optional): The maximum depth of the tree. Defaults to 10.
            min_samples_split (int, optional): The minimum number of samples to split a node. Defaults to 4.
            min_impurity (float, optional): The minimum impurity to split a node. Defaults to 0.1.
            impurity_measure (str): The impurity measure to use. Defaults to "gini".
        """

        self.X = X
        self.y = y
        self.root = Node(X, y, None, None, None, self, 0, "root")
        self.nodes = [self.root]

        # Set tree hyperparameters.
        self.impurity_measure = kwargs.get("impurity_measure", "gini")
        self.max_depth = kwargs.get("max_depth", 10)
        self.min_samples_split = kwargs.get("min_samples_split", 4)
        self.min_impurity = kwargs.get("min_impurity", 0.1)

        self.deepest_node = 0
        self._decision_nodes = [self.root]
        self._leaf_nodes = []
        self._features = []

        self.params = kwargs

    @property
    def features(self) -> list:

        if len(self._features) == 0:
            for node in self.decision_nodes:
                self._features.append(node.feature)

        return self._features

    @property
    def decision_nodes(self) -> list:

        # If tree has not been fitted then we have only the root node.
        # If it has been fitted there are decision nodes > 1
        if len(self._decision_nodes) == 1:
            for node in self.nodes:
                if node.__class__.__name__ in ["Node", "Root"]:
                    self._decision_nodes.append(node)
        return self._decision_nodes

    @property
    def leaf_nodes(self) -> list:

        # Do the same as for decision_nodes
        if len(self._leaf_nodes) == 0:
            for node in self.nodes:
                if node.__class__.__name__ == "Leaf":
                    self._leaf_nodes.append(node)
        return self._leaf_nodes

    def print_tree(self) -> None:
        """Print the whole generated tree following the path of generation.
        
        This method simply calls DecisionTree.print_children(root)
        """

        self.print_children(self.root)

    @staticmethod
    def print_children(node) -> None:
        """Print the children of the given node following the recursive path of generation of the tree.
        
        Args:
            node (Node): What node to start with.
        """

        print(node)
        if node.__class__.__name__ == "Leaf":
            return
        DecisionTree.print_children(node.left)
        DecisionTree.print_children(node.right)

    def feature_importance(self):
        """Calculate the feature importance of the tree.

        Returns:
            feature_importance (np.ndarray): The feature importance of the tree.
        """

        feature_importance = np.zeros(self.X.shape[1])

        all_nodes_importance = np.sum(list(map(lambda x: x.importance, self.decision_nodes)))

        for node in self.decision_nodes:
            feature_importance[node.feature] += node.importance / all_nodes_importance

        # Normalize the feature importance
        feature_importance /= np.sum(feature_importance)

        return feature_importance

    def append(self, node):
        """Append a node to the tree.

        Args:
            node (Node): The node to append to the tree.
        """

        self.nodes.append(node)

    def replace_node(self, node, leaf):
        """Replace a node with a leaf in the tree.

        Args:
            node (Node): The node to replace.
            leaf (Leaf): The leaf to replace the node with.
        """

        if node.parent is None:
            self.root = leaf
            return

        if node.parent.left == node:
            node.parent.left = leaf
        else:
            node.parent.right = leaf

        leaf.parent = node.parent
        self.nodes.remove(node)
        self.append(leaf)

        del node

        return leaf

    def __str__(self) -> str:
        """Return a string representation of the tree.

        Returns:
            str: The string representation of the tree.
        """

        return f"Tree(nodes={self.nodes})"

    def fit(self):
        """Fit the decision tree to the data."""
        
        self.root.train(max_depth=self.max_depth)

    def plot(self, path:Path, label_colors:list=["red", "green"]):
        """Plot the decision tree.
        
        Args:
            path (Path): The path to save the plot to.
            label_colors (list, optional): The colors to use for the labels. Defaults to ["red", "green"].
        """

        self.label_colors = label_colors

        # Set max right, left and depth for plot axes.
        max_r = 0
        max_l = 0
        max_d = 0

        # Call plot method of each node.
        for node in self.nodes:
            node.plot()

            # If node is leaf, update plot axes.
            if node.__class__.__name__ == "Leaf":
                if node.depth > max_d:
                    max_d = node.depth
                if node._path > max_r:
                    max_r = node._path
                if node._path < max_l:
                    max_l = node._path
                
        # construct a legend.
        legend = []
        for color, (label, _) in zip(self.label_colors, enumerate(self.root.bc)):
            legend.append(mpatches.Circle(color=color, label=label, xy=(0, 0), radius=1))

        # Insert root legend.
        legend.append(mpatches.Circle(color="yellow", label="root", xy=(0,0), radius=1))

        # Set dashed lines for each depth level.
        depth_lines = np.arange(self.max_depth)
        plt.hlines(depth_lines, -1.5, 1.5, colors="black", linestyles="dashed", alpha=0.3)

        # Set axes limits.
        plt.yticks(depth_lines, depth_lines)
        plt.xticks([])
        plt.xlim(max_l - 1, max_r + 1)
        plt.ylim(-1, max_d + 1)

        plt.legend(handles=legend)
        plt.gca().invert_yaxis()
        plt.title(f"Decision Tree with max_depth={self.max_depth}")
        plt.savefig(path)
        plt.clf()
        # plt.show()

    def see_leaves(self):
        """Print the leaves of the tree (meant for debugging)."""

        for node in self.nodes:
            if node.__class__.__name__ == "Leaf":
                print(node)

    @staticmethod
    def predict_sample(node, X:np.ndarray):
        """Predict the label of a sample or array of samples.

        Args:
            node (Node): The node to start the prediction from (generally use DecisionTree.root).
            X (np.ndarray): The sample to predict the label(s) of.

        Returns:
            int|np.ndarray: The predicted label(s).
        """

        # If predict struck a leaf, return the label of the leaf.
        if node.__class__.__name__ == "Leaf":
            return node.label

        if X[node.feature] <= node.threshold:
            return DecisionTree.predict_sample(node.left, X)

        return DecisionTree.predict_sample(node.right, X)

    def predict(self, X:np.ndarray) -> np.ndarray:
        """Predict label of X.
        
        Args:
            X (np.array): The samples to predict label of.
        
        Returns:
            predictions (np.array): The predictions of the classifier.
        """

        predictions = np.zeros_like(X[:, 0])

        for i, row in enumerate(X):
            predictions[i] = DecisionTree.predict_sample(self.root, row)
        
        return predictions

    def score(self, X:np.ndarray, y:np.ndarray):
        """Calculate the score of the tree.

        Args:
            X (np.ndarray): The data to score the tree on.
            y (np.ndarray): The labels to score the tree on.

        Returns:
            float: The score of the tree.
        """

        predictions = self.predict(X)

        return np.sum(predictions == y) / len(y)

    def predict_probas(self, X:np.ndarray) -> np.ndarray:
        """Predict the probabilities of the labels of X.
        
        Args:
            X (np.ndarray): The samples to predict the probabilities of.

        Returns:
            probas (np.ndarray): The probabilities of the labels.
        """

        probas = np.zeros((X.shape[0], len(self.root.bc)))

        for i, row in enumerate(X):
            probas[i] = DecisionTree.predict_sample_probas(self.root, row)

        return probas
    
    @staticmethod
    def predict_sample_probas(node, X:np.ndarray) -> np.ndarray:
        """Predict the probabilities of the labels of a sample.

        Args:
            node (Node): The node to start the prediction from (generally use DecisionTree.root).
            X (np.ndarray): The sample to predict the probabilities of.

        Returns:
            probas (np.ndarray): The probabilities of the labels.
        """

        # If predict struck a leaf, return the class probability of the leaf.
        if node.__class__.__name__ == "Leaf":
            return node.probas

        if X[node.feature] <= node.threshold:
            return DecisionTree.predict_sample_probas(node.left, X)

        return DecisionTree.predict_sample_probas(node.right, X)


class Leaf:
    """A leaf in the decision tree.

    Attributes:
        label (int): The label of the leaf.
    
    Methods:
        __init__(label): Initialize the leaf.
        __str__(): Return a string representation of the leaf.
    """

    path_vals = {"left":-1, "right":1, "root":0}

    def __init__(self, X:np.ndarray, y:np.ndarray, depth:int, direction:str, parent, tree) -> None:
        """Initialize the leaf.

        Args:
            label (int): The label of the leaf.
        """

        self.X = X
        self.y = y
        self.depth = depth
        self.direction = direction
        self.parent = parent
        self.tree = tree
        self.impurity = 0
        self.bc = np.bincount(self.y)
        self.samples = len(self.y)

        # Set leaf's label as the most common class
        self.label = self.bc.argmax()

        self._importance = 0
        self._probas = None

    @property
    def importance(self):
        """Return the importance of the leaf.

        Returns:
            float: The importance of the leaf.
        """

        return self._importance

    def __str__(self) -> str:
        """Return a string representation of the leaf.

        Returns:
            str: The string representation of the leaf.
        """

        # return f"Leaf(dir={self.direction}, depth={self.depth}, label={self.label}, bc={self.bc})"
        return f"Leaf(\n\tlabel={self.label},\n\tsamples={self.samples}\n\tvalues={self.bc}\n\tdepth={self.depth}\n)"


    def __repr__(self) -> str:
        """Return a string representation of the leaf.

        Returns:
            str: The string representation of the leaf.
        """

        # return f"Leaf(dir={self.direction}, depth={self.depth}, label={self.label}, bc={self.bc})"
        return f"Leaf(\n\tlabel={self.label},\n\tsamples={self.samples}\n\tvalues={self.bc}\n\tdepth={self.depth}\n)"

    def plot(self):
        """Plot the leaf."""

        self._path, parents_path = self.path()

        point_color = self.tree.label_colors[self.label] if self.depth != 0 else "yellow"

        plt.plot([self._path, parents_path], [self.depth, self.depth - 1 if self.depth != 0 else 0], color="blue")
        plt.scatter(self._path, self.depth, color=point_color, s=100)


    def path(self):
        """Recursively find the path to the root node."""
        
        if self.depth == 0:
            return 0, 0

        parents_path, _ = self.parent.path()

        return self.path_vals[self.direction] / self.depth**2 + parents_path, parents_path

    @property
    def probas(self):
        """Calculate the class probabilities of the leaf.

        Returns:
            np.ndarray: The class probabilities of the leaf.
        """

        if not self._probas is None:
            return self._probas

        if self.bc.shape[0] == 1:
            return np.array([1, 0])

        return self.bc / self.samples


class Node(Leaf):
    """A node in the decision tree.

    Attributes:
        feature (int): The feature that the node splits on.
        threshold (float): The threshold that the node splits on.
        left (Node): The left child of the node.
        right (Node): The right child of the node.
        label (int): The label of the node.
    
    Methods:
        __init__(feature, threshold, left, right, label): Initialize the node.
        __str__(): Return a string representation of the node.
    """

    def __init__(self, X:np.ndarray, y:np.ndarray, feature:int, threshold:float, parent, tree, depth:int, direction) -> None:
        """Initialize the node.

        Args:
            feature (int): The feature that the node splits on.
            threshold (float): The threshold that the node splits on.
            left (Node|Leaf): The left child of the node.
            right (Node|Leaf): The right child of the node.
        """
        super().__init__(X, y, depth, direction, parent, tree)

        self.feature = feature
        self.threshold = threshold
        self.impurity = self.gini()
        self._importance = None

    def __str__(self) -> str:
        """Return a string representation of the node.

        Returns:
            str: The string representation of the node.
        """

        return f"{self.__class__.__name__}(\n\tX[{self.feature}] <= {self.threshold:.3f}\n\timportance={self.importance:.3f}\n\tgini={self.impurity:.3f}\n\tsamples={self.samples}\n\tvalues={self.bc}\n\tdepth={self.depth}\n)"

    def __repr__(self) -> str:
        """Return a string representation of the node.

        Returns:
            str: The string representation of the node.
        """

        return f"{self.__class__.__name__}(\n\tX[{self.feature}] <= {self.threshold:.3f}\n\timportance={self.importance:.3f}\n\tgini={self.impurity:.3f}\n\tsamples={self.samples}\n\tvalues={self.bc}\n\tdepth={self.depth}\n)"

    def __train_valid_node(self, max_depth):
        """Method for validating if recursion is to be continued or not.
        
        We use the basic criteria of if the node is a leaf, then break recursion, else continue.
        """

        instance = (isinstance(self.left, Node), isinstance(self.right, Node))

        match instance:

            # Both children are nodes
            case (True, True):

                # Trying some semi-sketchy multiprocessing here.
                if self.__class__.__name__ == "Root":
                    with Pool(processes=2) as pool:
                        left, right = pool.map(
                            lambda x: x.train(max_depth), 
                            [self.left, self.right]
                        )
                    return left, right
                return self.left.train(max_depth), self.right.train(max_depth)

            # Left child is a node
            case (True, False):
                return self.left.train(max_depth), self.right

            # Right child is a node
            case (False, True):
                return self.left, self.right.train(max_depth)
            
            # Both children are leaves
            case (False, False):
                return self.left, self.right

    def train(self, max_depth):
        """Train the root node.

        Args:
            max_depth (int): The maximum depth of the tree.

        Returns:
            Node: The root node of the tree.
        """

        result = self.best_split()

        if result.__class__.__name__ == "Leaf" or result is None:
            return result
        
        left, right = result
        
        for node in [left,right]:
            declare_leaf = False

            if self.gain < self.tree.min_impurity:
                declare_leaf = True

            if node is None:
                print(node)

            if node.depth == max_depth:
                declare_leaf = True
            
            if len(np.unique(node.y)) == 1:
                declare_leaf = True

            if len(node.y) <= self.tree.min_samples_split:
                declare_leaf = True

            if declare_leaf:
                node = Leaf(node.X, node.y, node.depth, node.direction, self, self.tree)
                self.tree.deepest_node = max(self.tree.deepest_node, node.depth)

            if node.direction == "left":
                self.left = node
            elif node.direction == "right":
                self.right = node

            self.tree.append(node)

        return self.__train_valid_node(max_depth)
        
    def entropy(self):
        """Calculate the entropy of the node.

        Returns:
            float: The entropy of the node.
        """

        # Get the number of samples in each class
        class_counts = np.bincount(self.y)

        # find each class probability
        probabilities = class_counts / self.y.shape[0]

        # Calculate the entropy with formula H = -sum(p * log2(p)), but p = 0 is undefined therefore remove it from calculation.
        probabilities = probabilities[probabilities > 0]
        entropy = -np.sum(probabilities * np.log2(probabilities))

        return entropy

    def gini(self):
        """Calculate the gini impurity of the node.

        Returns:
            float: The gini impurity of the node.
        """

        # Get the number of samples in each class
        class_counts = np.bincount(self.y)

        # find each class probability
        probabilities = class_counts / self.y.shape[0]

        # Calculate the gini impurity with formula G = 1 - sum(p**2)
        gini = 1 - np.sum(probabilities ** 2)

        return gini

    def information_gain(self, left, right, **kwargs):
        """Calculate the information gain of the node.

        Args:
            left (Node): The left child of the node.
            right (Node): The right child of the node.
        
        Keyword Args:
            impurity_measure (str): The impurity measure to use. Defaults to "gini".

        Returns:
            float: The information gain of the node.
        """

        # Get the impurity measure
        impurity_measure = kwargs.get("impurity_measure", "gini")

        this_impurity = getattr(self, impurity_measure)()
        left_impurity = getattr(left, impurity_measure)()
        right_impurity = getattr(right, impurity_measure)()

        # Calculate the information gain
        p = len(left.y) / len(self.y)
        information_gain = this_impurity - p * left_impurity - (1 - p) * right_impurity

        return information_gain

    def best_split(self):
        """Find the best split for the node.

        Returns:
            Node: The node with the best split.
        """

        # Variables tracking best split information
        best_information_gain = 0
        best_feature = None
        best_threshold = None
        best_left = None
        best_right = None

        # iterate over each feature
        for i, feature in enumerate(range(self.X.shape[1])):

            # set current feature
            feature_values = self.X[:, feature]

            # Thresholds may be any unique value, we dont bother splitting in between thresholds, but rather right on one.
            thresholds = np.unique(feature_values)

            # Iterate over each threshold
            for j, threshold in enumerate(thresholds):

                # Get left and right indices for each threshold
                left_indices, right_indices = self.split(feature, threshold)

                # Check if any of the indices are empty, then we wont bother calculating information gain.
                if len(left_indices) == 0 or len(right_indices) == 0:
                    
                    # If we are at the last threshold, then we have no choice but to split here.
                    # if j == len(thresholds) - 1 and i == self.X.shape[1] - 1:
                        

                    continue

                # Now construct a node for each split, this is currently just a test node, and will be discarded if it is not the best split.
                left = Node(self.X[left_indices], self.y[left_indices], feature, threshold, self, self.tree, self.depth + 1, "left")
                right = Node(self.X[right_indices], self.y[right_indices], feature, threshold, self, self.tree, self.depth + 1, "right")

                # Calculate the information gain for the current split
                information_gain = self.information_gain(left, right, impurity_measure=self.tree.impurity_measure)

                # store the best split
                if information_gain > best_information_gain:
                    best_left = left
                    best_right = right
                    best_threshold = threshold
                    best_feature = feature
                    best_information_gain = information_gain
        
        # Set nodes attributes
        self.feature = best_feature
        self.threshold = best_threshold
        self.gain = best_information_gain

        # If no split was found, then we are at a leaf node.
        if best_feature is None:
            leaf = Leaf(self.X, self.y, self.depth, self.direction, self.parent, self.tree)
            return self.tree.replace_node(self, leaf)

        # Return the left and right nodes
        return best_left, best_right

    def split(self, feature, threshold):
        """Split the node on the given feature and threshold.

        Args:
            feature (int): The feature to split on.
            threshold (float): The threshold to split on.

        Returns:
            tuple: The indices of the left and right child.
        """
        
        # get the index of the feature which is over/under the treshold
        left_indices = np.argwhere(self.X[:, feature] <= threshold).flatten()
        right_indices = np.argwhere(self.X[:, feature] > threshold).flatten()

        return left_indices, right_indices

    def declare_child_leaf(self, child):
        """Declare the child node as a leaf.

        Args:
            child (Node): The child node to declare as a leaf.

        Returns:
            Leaf: The leaf node.
        """

        if child.direction == "left":
            self.left = Leaf(child.X, child.y, child.depth, child.direction, self, self.tree)
            return self.left
        elif child.direction == "right":
            self.right = Leaf(child.X, child.y, child.depth, child.direction, self, self.tree)
            return self.right

    @property
    def importance(self):
        """Calculate the importance of the node.
        
        This method is set as a property indicating it can only be reached once the tree has been trained.

        Returns:
            float: The importance of the node.
        """

        if self._importance is None:
            # Get the total number of samples in the tree
            total_samples = self.tree.root.y.shape[0]

            # Get the number of samples in the node
            node_samples = self.y.shape[0]

            # Calculate the importance (has used that leaf importances are 0)
            importance = (node_samples / total_samples) * self.impurity - self.left.importance - self.right.importance
            
            # Update node importance so we dont have to calculate it again
            self._importance = importance
            
            return importance

        return self._importance

class Root(Node):
    def __init__(self, X, y, tree):
        super().__init__(X, y, None, None, None, tree, 0, None)
