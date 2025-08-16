import numpy as np

class AdalineGD:
    """
    ADAptive LInear NEuron classifier (using batch gradient descent).

    Parameters
    ----------
    lr : float
        Learning rate (0.0 - 1.0).
    epochs : int
        Number of passes over the training dataset.
    seed : int
        Random seed for reproducible weight initialization.

    Attributes
    ----------
    weights : 1D numpy array
        Model weights (bias is stored as weights[0]).
    costs : list
        Sum of squared errors for each epoch (used to check convergence).
    """

    def __init__(self, lr=0.01, epochs=50, seed=1):
        self.lr = lr
        self.epochs = epochs
        self.seed = seed

    def fit(self, X, y):
        """
        Train the Adaline model using batch gradient descent.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training inputs.
        y : array-like, shape = [n_samples]
            Target labels (-1 or 1).

        Returns
        -------
        self : object
        """
        rng = np.random.RandomState(self.seed)
        # initialize weights (including bias as first element)
        self.weights = rng.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        self.costs = []

        for epoch in range(self.epochs):
            net_out = self.net_input(X)
            output = self.activation(net_out)  # identity here
            errors = (y - output)

            # update weights (batch update)
            self.weights[1:] += self.lr * X.T.dot(errors)
            self.weights[0] += self.lr * errors.sum()

            # compute cost (SSE)
            cost = (errors**2).sum() / 2.0
            self.costs.append(cost)
        return self

    def net_input(self, X):
        # Linear combination: w.x + b
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def activation(self, X):
        # Activation function: here it’s just the identity.
        return X

    def predict(self, X):
        # Class label: apply threshold at 0.0
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
