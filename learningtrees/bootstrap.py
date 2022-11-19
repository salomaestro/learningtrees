import numpy as np

class BootstrapData:
    """Generate Bootstrap data for a given data set

    Attributes:
        X: data set
        n: number of samples to use in bootstrapped data set
        max_bootstraps: maximum number of bootstraps to use, defaults
        sample_matrix: matrix of indices for each bootstrap sample
        OOB_test_matrix: matrix of indices for each OOB test

    Methods:
        _generate_OOB_matrix: generate OOB test matrix
    """
    def __init__(self, X, **kwargs):
        """Initialize BootstrapData object
        
        Args:
            X: data set
        
        Keyword Args:
            max_bootstraps (int): maximum number of bootstraps to use, defaults to length of samples in dataset - 1
            n_samples (int): numer of samples to draw (with replacement) from X along axis 0. Defaults to length - 1 of X along axis 0.
        """

        self.X = X
        self.n = self.X.shape[0] - 1

        self.max_bootstraps = min(kwargs.get('max_bootstraps', X.shape[0]-1), X.shape[0]-1)
        
        if kwargs["n_samples"]:
            self.n = min(kwargs.get("n_samples", self.n), self.n)

        self.sample_matrix = np.random.choice(self.X.shape[0], size=(self.n, self.max_bootstraps), replace=True)
        # self.OOB_test_matrix = self._generate_OOB_matrix(self.sample_matrix)

    @staticmethod
    def _generate_OOB_matrix(sample_matrix):
        """Generate a matrix which keeps track column wise if a sample can be used for out of bag error testing.
        
        Each column represent a subset, where the index correspond to the index of the dataset. If the value is 1,
        it means the sample have not been used in the subset matrix and can therefore instead be used for testing.

        Args:
            sample_matrix (np.array): n x n-1 matrix, of samples indices.
        """
        
        # Set up an empty array for storing results in
        res = np.zeros_like(sample_matrix)

        # iterate over each column of the sample_matrix
        for i, subset in enumerate(sample_matrix.T):

            # If a sample have not been sampled in its respective subset matrix we give it a 1, to keep track of
            # what samples to use for Out of Bag testing.
            res.T[i] += np.where(
                np.bincount(subset, minlength=len(subset)) > 0, 0, 1
            )

        return res

if __name__ == "__main__":
    data = np.arange(9**2).reshape((9,9))

    np.random.seed(42)

    b1 = BootstrapData(data, max_n=5)
    b2 = BootstrapData(data, max_n=20)

    assert b1.n == 5
    assert b2.n == data.shape[0] - 1

    mat = b1.sample_matrix

    OOB = b1.OOB_test_matrix