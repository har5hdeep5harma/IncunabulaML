import numpy as np
from incunabula.perceptron import Perceptron

def test_perceptron_training():
    # Simple linearly separable dataset (AND logic gate style)
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([-1, -1, -1, 1]) 

    model = Perceptron(lr=0.1, epochs=10, seed=1)
    model.fit(X, y)

    preds = model.predict(X)

    # Model should learn this perfectly
    assert (preds == y).all()

def test_perceptron_weights_not_empty():
    X = np.array([[0, 0], [1, 1]])
    y = np.array([-1, 1])

    model = Perceptron()
    model.fit(X, y)

    # Ensuring weights got initialized and updated
    assert model.weights is not None
    assert len(model.weights) == X.shape[1] + 1
