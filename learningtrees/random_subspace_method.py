import numpy as np

from util.decisiontrees.bootstrap import BootstrapData


class RandomSubspaceMethod:
    """Use the Random subspace method to generate sample and feature matricies indicating
    what data to use.
    
    Attributes:
        sample_matrix: matrix of samples to use, as an (n x m) matrix, where columns values indicate indexes of samples to use,
            and different columns are for different trees.
        feature_matrix: matrix of features to use as an (p x m) matrix, where column values indicate indexes of features to use,
            and different columns are for different trees.
    """

    def __init__(self, X, n_subsets=1, **kwargs):
        """Initialize random subset method.

        Args:
            X (np.ndarray): dataset
            n_subsets (Optional, int): how many different trees will be sampled for. Defaults to 1.
        
        Keyword Args:
            n_samples (int): number of samples to draw (with replacement) from X along axis 0. Defaults to length - 1 of X along axis 0.
            n_features (int): number of features to draw (with replacement) from X along axis 1. Default to length - 1 of X along axis 1.
        """

        n_samples = min(kwargs.get("n_samples", X.shape[0] - 1), X.shape[0] - 1)
        n_features = min(kwargs.get("n_features", X.shape[1] - 1), X.shape[1] - 1)

        self._sample_matrix = BootstrapData(X, max_bootstraps=n_subsets, n_samples=n_samples).sample_matrix

        self._feature_matrix = np.random.choice(X.shape[1], size=(n_features, n_subsets))

    @property
    def sample_matrix(self):
        return self._sample_matrix

    @property
    def feature_matrix(self):
        return self._feature_matrix

if __name__ == "__main__":
    X = np.random.randint(-10, 10, size=(10, 13))

    subset = RandomSubspaceMethod(X, n_subsets=4, n_samples=8, n_features=3)

    print(subset.sample_matrix)
    print(subset.feature_matrix)
