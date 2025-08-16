import numpy as np

class Perceptron:
    """
    Simple Perceptron classifier.

    Parameters
    ----------
    lr : float
        Learning rate (0.0 - 1.0), controls the step size in updates.
    epochs : int
        Number of passes over the training data.
    seed : int
        Random seed for reproducible weight initialization.

    Attributes
    ----------
    weights : 1D numpy array
        Model weights (bias is stored as weights[0]).
    mistakes_per_epoch : list
        Number of misclassifications in each epoch.
    """

    def __init__(self, lr=0.01, epochs=50, seed=1):
        self.lr = lr
        self.epochs = epochs
        self.seed = seed

    def fit(self, X, y):
        #Train the perceptron model.
        rng = np.random.RandomState(self.seed)
        # +1 for bias weight
        self.weights = rng.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.mistakes_per_epoch = []

        for epoch in range(self.epochs):
            mistakes = 0
            for xi, target in zip(X, y):
                update = self.lr * (target - self.predict(xi))
                self.weights[1:] += update * xi   
                self.weights[0] += update  
                mistakes += int(update != 0.0)
            self.mistakes_per_epoch.append(mistakes)
        return self

    def net_input(self, X):
        #Linear combination: w.x + b
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        #Unit step function: returns -1 or 1
        return np.where(self.net_input(X) >= 0.0, 1, -1)
