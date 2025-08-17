import numpy as np
import pytest
from incunabula.rbf_kernel_pca import rbf_kernel_pca

def test_rbf_kernel_pca_shapes():
    # Toy dataset: circle-like
    X = np.array([[1, 0], [0, 1], [-1, 0], [0, -1],
                  [0.7, 0.7], [-0.7, -0.7]])

    gamma = 10
    n_components = 2

    alphas, lambdas = rbf_kernel_pca(X, gamma=gamma, n_components=n_components)

    # Check shapes
    assert alphas.shape == (X.shape[0], n_components)
    assert len(lambdas) == n_components

    # Eigenvalues should be non-negative
    for val in lambdas:
        assert val >= 0

def test_rbf_kernel_pca_single_component():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    gamma = 3
    n_components = 1

    alphas, lambdas = rbf_kernel_pca(X, gamma=gamma, n_components=n_components)

    # Should return only 1D projections
    assert alphas.shape[1] == 1
    assert len(lambdas) == 1
