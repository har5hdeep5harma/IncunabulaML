import numpy as np
import pytest
from incunabula.neuralnet import NeuralNetMLP


def test_forward_pass_output_shape():
    # Check that forward propagation produces the correct output shape.
    nn = NeuralNetMLP(n_features=4, n_hidden=5, n_output=3, random_state=1)
    X_sample = np.random.randn(10, 4)  # 10 samples, 4 features
    _, _, _, output = nn._forward(nn._add_bias_unit(X_sample, how='column'))
    assert output.shape == (3, 10)  # 3 classes, 10 samples


def test_prediction_shape():
    # Check that predictions return the correct number of labels.
    nn = NeuralNetMLP(n_features=4, n_hidden=5, n_output=3, random_state=1)
    X_sample = np.random.randn(8, 4)
    y_pred = nn.predict(X_sample)
    assert y_pred.shape == (8,)  # one predicted label per sample


def test_fit_reduces_cost():
    # Ensure that training reduces the cost over epochs.
    # small dataset
    X_train = np.array([[0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1]])
    y_train = np.array([0, 1, 1, 0])  # XOR labels

    nn = NeuralNetMLP(n_features=2, n_hidden=4, n_output=2,
                      epochs=10, eta=0.1, random_state=1)

    nn.fit(X_train, y_train)

    assert len(nn.cost_) == 10
    assert nn.cost_[0] > nn.cost_[-1]  # cost should decrease


def test_onehot_encoding():
    # Check one-hot encoding correctness.
    nn = NeuralNetMLP(n_features=2, n_hidden=2, n_output=3)
    labels = np.array([0, 1, 2])
    onehot = nn._onehot(labels, n_classes=3)
    expected = np.array([[1., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.]])
    assert np.array_equal(onehot, expected)


def test_invalid_bias_mode():
    # Ensure invalid bias unit mode raises ValueError.
    nn = NeuralNetMLP(n_features=2, n_hidden=2, n_output=2)
    with pytest.raises(ValueError):
        _ = nn._add_bias_unit(np.array([[1, 2], [3, 4]]), how='invalid')
