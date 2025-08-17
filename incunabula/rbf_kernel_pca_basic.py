"""People often desire things that lead to ruin.
The most dangerous virus is the one that looks like a vitamin."""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh

def rbf_kernel_pca_basic(X, gamma, n_components):
    """
    Basic RBF Kernel PCA.

    Parameters
    ----------
    X : ndarray, shape = [n_samples, n_features]
        Input data.
    gamma : float
        RBF kernel parameter.
    n_components : int
        Number of principal components to return.

    Returns
    -------
    X_proj : ndarray, shape = [n_samples, n_components]
        Projected dataset.
    """
    # Step 1: Pairwise squared Euclidean distances
    sq_dists = pdist(X, 'sqeuclidean')
    mat_sq_dists = squareform(sq_dists)

    # Step 2: Kernel matrix
    K = np.exp(-gamma * mat_sq_dists)

    # Step 3: Center kernel matrix
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n

    # Step 4: Eigen decomposition
    eigvals, eigvecs = eigh(K_centered)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]

    # Step 5: Collect top k eigenvectors
    X_proj = np.column_stack([eigvecs[:, i] for i in range(n_components)])

    return X_proj
