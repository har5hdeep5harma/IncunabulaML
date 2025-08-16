import numpy as np
from incunabula.adaline_gd import AdalineGD

def test_adalinegd_training():
    # Simple linearly separable dataset (AND-like)
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([-1, -1, -1, 1])

    model = AdalineGD(lr=0.1, epochs=20, seed=1)
    model.fit(X, y)

    preds = model.predict(X)

    # Should classify correctly
    assert (preds == y).all()

def test_adalinegd_costs_decreasing():
    X = np.array([[0], [1]])
    y = np.array([-1, 1])

    model = AdalineGD(lr=0.01, epochs=10, seed=1)
    model.fit(X, y)

    # Check that costs got recorded
    assert len(model.costs) == 10

    # Cost should generally decrease (not strictly monotonic, but end < start)
    assert model.costs[-1] <= model.costs[0]

def test_adalinegd_weights_shape():
    X = np.array([[0, 0], [1, 1]])
    y = np.array([-1, 1])

    model = AdalineGD(lr=0.1, epochs=5, seed=1)
    model.fit(X, y)

    # Ensure weights exist and match features + bias
    assert model.weights is not None
    assert len(model.weights) == X.shape[1] + 1
