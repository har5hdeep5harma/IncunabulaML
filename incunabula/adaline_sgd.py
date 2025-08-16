import numpy as np

class AdalineSGD:
    """
    ADAptive LInear NEuron classifier (using stochastic gradient descent).

    Parameters
    ----------
    lr : float
        Learning rate (0.0 - 1.0).
    epochs : int
        Number of passes over the training dataset.
    shuffle : bool
        If True, shuffle training data every epoch to avoid cycles.
    seed : int
        Random seed for reproducible results.

    Attributes
    ----------
    weights : 1D numpy array
        Model weights (bias is stored as weights[0]).
    costs : list
        Cost (SSE) after each training epoch.
    """

    def __init__(self, lr=0.01, epochs=50, shuffle=True, seed=None):
        self.lr = lr
        self.epochs = epochs
        self.shuffle = shuffle
        self.seed = seed
        self.weights_initialized = False

    def fit(self, X, y):
        #Train the Adaline model using stochastic gradient descent.
        self._initialize_weights(X.shape[1])
        self.costs = []

        for epoch in range(self.epochs):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.costs.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        # Train model without reinitializing weights (useful for online learning).
        if not self.weights_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        # Shuffle training data each epoch.
        rng = np.random.RandomState(self.seed)
        perm = rng.permutation(len(y))
        return X[perm], y[perm]

    def _initialize_weights(self, n_features):
        # Randomly initialize weights.
        rng = np.random.RandomState(self.seed)
        self.weights = rng.normal(loc=0.0, scale=0.01, size=1 + n_features)
        self.weights_initialized = True

    def _update_weights(self, xi, target):
        # Apply SGD weight update for one sample.
        net_out = self.net_input(xi)
        output = self.activation(net_out)
        error = (target - output)

        self.weights[1:] += self.lr * xi * error
        self.weights[0] += self.lr * error

        cost = 0.5 * (error**2)
        return cost

    def net_input(self, X):
        # Linear combination: w.x + b
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def activation(self, X):
        # Identity activation (linear).
        return X

    def predict(self, X):
        # Return class label after thresholding at 0.0
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
