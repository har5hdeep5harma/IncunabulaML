import numpy as np
from incunabula.adaline_sgd import AdalineSGD

def test_adalinesgd_training():
    # Simple AND-like dataset
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([-1, -1, -1, 1])

    model = AdalineSGD(lr=0.1, epochs=20, seed=1, shuffle=True)
    model.fit(X, y)

    preds = model.predict(X)

    # Should learn perfectly
    assert (preds == y).all()

def test_adalinesgd_costs_recorded():
    X = np.array([[0], [1]])
    y = np.array([-1, 1])

    model = AdalineSGD(lr=0.01, epochs=5, seed=1)
    model.fit(X, y)

    # Ensure cost list was populated
    assert len(model.costs) == 5

def test_adalinesgd_partial_fit():
    X = np.array([[0], [1]])
    y = np.array([-1, 1])

    model = AdalineSGD(lr=0.1, epochs=1, seed=1)
    model.partial_fit(X, y)

    # After partial fit, weights should be initialized
    assert model.weights is not None
    assert len(model.weights) == X.shape[1] + 1

def test_adalinesgd_weights_shape():
    X = np.array([[0, 0], [1, 1]])
    y = np.array([-1, 1])

    model = AdalineSGD(lr=0.1, epochs=5, seed=1)
    model.fit(X, y)

    # Check correct number of weights
    assert len(model.weights) == X.shape[1] + 1
