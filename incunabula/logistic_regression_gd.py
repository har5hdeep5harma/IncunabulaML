import numpy as np

class LogisticRegressionGD:
    """
    Logistic Regression classifier trained with gradient descent.

    Parameters
    ----------
    lr : float
        Learning rate (between 0.0 and 1.0).
    epochs : int
        Number of passes over the training data.
    seed : int
        Random number generator seed for reproducible weight initialization.

    Attributes
    ----------
    weights : 1d-array
        Final weight vector after training.
    losses : list
        Cost function value in each epoch.
    """

    def __init__(self, lr=0.05, epochs=100, seed=1):
        self.lr = lr
        self.epochs = epochs
        self.seed = seed

    def fit(self, X, y):
        # Train the model on the dataset X, y.
        rng = np.random.RandomState(self.seed)
        self.weights = rng.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.losses = []

        for epoch in range(self.epochs):
            net_out = self._net_input(X)
            prob = self._sigmoid(net_out)
            errors = y - prob

            # update rule
            self.weights[1:] += self.lr * X.T.dot(errors)
            self.weights[0] += self.lr * errors.sum()

            # logistic loss (cross-entropy)
            cost = - (y.dot(np.log(prob)) + (1 - y).dot(np.log(1 - prob)))
            self.losses.append(cost)

        return self

    def _net_input(self, X):
        # Linear combination of inputs and weights.
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def _sigmoid(self, z):
        # Logistic sigmoid activation, clipped for numerical stability.
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        # Class label prediction (0 or 1).
        return np.where(self._net_input(X) >= 0.0, 1, 0)
