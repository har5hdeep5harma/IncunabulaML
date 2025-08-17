import numpy as np
import pytest
from incunabula.rbf_kernel_pca_basic import rbf_kernel_pca_basic

def test_rbf_kernel_pca_basic_shape():
    # Toy dataset: 6 points in 2D
    X = np.array([[1, 0], [0, 1], [-1, 0], [0, -1],
                  [0.5, 0.5], [-0.5, -0.5]])

    gamma = 15
    n_components = 2

    X_proj = rbf_kernel_pca_basic(X, gamma=gamma, n_components=n_components)

    # Check shape of result
    assert X_proj.shape == (X.shape[0], n_components)

    # Should not be all zeros
    assert not np.allclose(X_proj, 0)

def test_rbf_kernel_pca_basic_more_components_than_features():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    gamma = 5
    n_components = 1

    X_proj = rbf_kernel_pca_basic(X, gamma=gamma, n_components=n_components)

    # If n_components = 1, we should only get one dimension
    assert X_proj.shape[1] == 1
