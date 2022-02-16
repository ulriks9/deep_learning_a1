from scipy.special import expit
import numpy as np

class ANN:

    def __init__(self, act='sigmoid', loss='SSE', n_features=2):
        self.weights = []
        self.biases = []
        self.act = act
        self.n_units = []
        self.n_units.append(n_features)

    def add_hidden(self, n_units):
        self.weights.append(np.random.uniform(low=-1, high=1, size=(n_units, self.n_units[-1])))
        self.biases.append(np.random.random(n_units))
        self.n_units.append(n_units)

    def forward_pass(self, x):
        for w_matrix, b_matrix in zip(self.weights, self.biases):
            x = np.dot(x, np.transpose(w_matrix)) + b_matrix

            if self.act == 'sigmoid':
                x = expit(x)

        return x

    def predict(self, x):
        return np.argmax(self.forward_pass(x))

