"""
NeuralNetMLP
------------
A simple implementation of a Multi-Layer Perceptron (MLP) neural network
trained with stochastic gradient descent (SGD).

This version is written for educational purposes and does not rely on 
high-level deep learning libraries like TensorFlow or PyTorch.
"""

import numpy as np


class NeuralNetMLP:
    """
    A simple multi-layer perceptron (feedforward neural network).

    Parameters
    ----------
    n_features : int
        Number of input features.
    n_hidden : int
        Number of units in the hidden layer.
    n_output : int
        Number of output units (classes).
    l1 : float, default=0.0
        L1 regularization strength.
    l2 : float, default=0.0
        L2 regularization strength.
    epochs : int, default=100
        Number of passes through the training dataset.
    eta : float, default=0.01
        Learning rate (step size).
    alpha : float, default=0.0
        Momentum constant (0 disables momentum).
    decrease_const : float, default=0.0
        Factor to decrease learning rate over epochs.
    shuffle : bool, default=True
        Whether to shuffle training data each epoch.
    minibatches : int, default=1
        Number of minibatches for SGD (1 means batch gradient descent).
    random_state : int, default=None
        Random seed for reproducibility.
    """

    def __init__(self, n_features, n_hidden, n_output,
                 l1=0.0, l2=0.0,
                 epochs=100, eta=0.01, alpha=0.0,
                 decrease_const=0.0, shuffle=True,
                 minibatches=1, random_state=None):

        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha
        self.decrease_const = decrease_const
        self.shuffle = shuffle
        self.minibatches = minibatches
        self.random_state = random_state

        # Initialize weights
        rgen = np.random.RandomState(self.random_state)
        self.w_h = rgen.normal(loc=0.0, scale=0.1,
                               size=(self.n_hidden, self.n_features + 1))
        self.w_out = rgen.normal(loc=0.0, scale=0.1,
                                 size=(self.n_output, self.n_hidden + 1))

    def _onehot(self, y, n_classes):
        # Convert class labels to one-hot encoded format.
        onehot = np.zeros((len(y), n_classes))
        for idx, val in enumerate(y.astype(int)):
            onehot[idx, val] = 1.0
        return onehot

    def _sigmoid(self, z):
        # Sigmoid activation function.
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def _sigmoid_derivative(self, z):
        # Derivative of sigmoid function (for backpropagation).
        return self._sigmoid(z) * (1.0 - self._sigmoid(z))

    def _add_bias_unit(self, X, how='column'):
        # Add bias unit (1) to array.
        if how == 'column':
            return np.c_[np.ones((X.shape[0], 1)), X]
        elif how == 'row':
            return np.r_[[np.ones((1, X.shape[1]))], X]
        else:
            raise ValueError("`how` must be 'column' or 'row'")

    def _forward(self, X):
        """
        Forward propagation.
        
        Parameters
        ----------
        X : ndarray, shape = [n_samples, n_features]
            Input feature matrix.

        Returns
        -------
        z_h, a_h, z_out, a_out : arrays
            Linear and activated outputs at hidden and output layers.
        """
        # Hidden layer
        z_h = self.w_h.dot(X.T)
        a_h = self._sigmoid(z_h)

        # Add bias to hidden activations
        a_h = self._add_bias_unit(a_h, how='row')

        # Output layer
        z_out = self.w_out.dot(a_h)
        a_out = self._sigmoid(z_out)

        return z_h, a_h, z_out, a_out

    def _compute_cost(self, y_enc, output):
        #Compute logistic cost with L1 and L2 regularization.
        L1_term = (self.l1 *
                   (np.abs(self.w_h[:, 1:]).sum() +
                    np.abs(self.w_out[:, 1:]).sum()))
        L2_term = (self.l2 *
                   (np.square(self.w_h[:, 1:]).sum() +
                    np.square(self.w_out[:, 1:]).sum()))

        # Cross-entropy cost
        term1 = -y_enc * (np.log(output.T + 1e-5))
        term2 = (1 - y_enc) * np.log(1 - output.T + 1e-5)
        cost = np.sum(term1 - term2) + L1_term + L2_term
        return cost

    def predict(self, X):
        # Predict class labels for given samples.
        _, _, _, a_out = self._forward(self._add_bias_unit(X, how='column'))
        return np.argmax(a_out, axis=0)

    def fit(self, X, y):
        """
        Fit the model using stochastic gradient descent.

        Parameters
        ----------
        X : ndarray, shape = [n_samples, n_features]
            Training feature matrix.
        y : ndarray, shape = [n_samples]
            Target labels.

        Returns
        -------
        self
        """
        self.cost_ = []
        y_enc = self._onehot(y, self.n_output)

        delta_w_h_prev = np.zeros(self.w_h.shape)
        delta_w_out_prev = np.zeros(self.w_out.shape)

        for epoch in range(self.epochs):

            # adaptive learning rate
            self.eta /= (1.0 + self.decrease_const * epoch)

            if self.shuffle:
                idx = np.arange(X.shape[0])
                np.random.shuffle(idx)
                X, y_enc = X[idx], y_enc[idx]

            mini = np.array_split(range(y.shape[0]), self.minibatches)
            for idx in mini:
                # Forward pass
                z_h, a_h, z_out, a_out = self._forward(
                    self._add_bias_unit(X[idx], how='column'))

                # Backpropagation
                delta_out = a_out - y_enc[idx].T
                delta_h = (self.w_out[:, 1:].T.dot(delta_out) *
                           self._sigmoid_derivative(z_h))

                # Gradient
                grad_w_out = delta_out.dot(a_h.T)
                grad_w_h = delta_h.dot(self._add_bias_unit(X[idx], how='column'))

                # Regularization
                grad_w_out[:, 1:] += self.l2 * self.w_out[:, 1:]
                grad_w_h[:, 1:] += self.l2 * self.w_h[:, 1:]

                # Weight update with momentum
                delta_w_h = (self.eta * grad_w_h) + (self.alpha * delta_w_h_prev)
                delta_w_out = (self.eta * grad_w_out) + (self.alpha * delta_w_out_prev)
                self.w_h -= delta_w_h
                self.w_out -= delta_w_out
                delta_w_h_prev, delta_w_out_prev = delta_w_h, delta_w_out

            # Compute cost per epoch
            z_h, a_h, z_out, a_out = self._forward(
                self._add_bias_unit(X, how='column'))
            cost = self._compute_cost(y_enc, a_out)
            self.cost_.append(cost)

        return self
