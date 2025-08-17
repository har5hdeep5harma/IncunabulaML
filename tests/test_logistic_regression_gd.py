import numpy as np
import pytest
from incunabula.logistic_regression_gd import LogisticRegressionGD

def test_logistic_regression_gd_fit_and_predict():
    # Toy dataset (OR gate style)
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([0, 0, 0, 1])  # Only [1,1] is positive

    model = LogisticRegressionGD(lr=0.1, epochs=50, seed=1)
    model.fit(X, y)

    # Check that weights were learned
    assert model.weights is not None
    assert len(model.losses) == 50

    # Predict on training data
    preds = model.predict(X)

    # Model should correctly classify most points
    assert np.array_equal(preds, y) or np.mean(preds == y) >= 0.75

def test_predict_output_shape():
    X = np.array([[0.5, 0.2], [1.2, -0.3]])
    y = np.array([0, 1])

    model = LogisticRegressionGD(lr=0.1, epochs=10, seed=1)
    model.fit(X, y)
    
    preds = model.predict(X)

    # Predictions should match number of samples
    assert preds.shape[0] == X.shape[0]
    # Predictions must be binary
    assert set(preds).issubset({0, 1})
