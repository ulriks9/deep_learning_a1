from scipy.special import expit, logit
import numpy as np
import sys

class ANN:

    def __init__(self, act='sigmoid', loss='SSE', n_features=2):
        self.weights = []
        self.biases = []
        self.act = act
        self.n_units = []
        self.n_units.append(n_features)
        self.loss = loss
        self.act = act

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

    def activations_forward_pass(self,x):
        A = []
        Z = []
        for w_matrix, b_matrix in zip(self.weights, self.biases):
            x = np.dot(x, np.transpose(w_matrix)) + b_matrix
            Z.append(x)
            if self.act == 'sigmoid':
                x = expit(x)
            A.append(x)
        return A,Z

    def backprop(self,A,Z,L,bias_changes,weight_changes,error):
        if L == 0:
            return

        n_units_L = len(self.biases[L])
        n_units_previous = len(self.biases[L-1])#

        error_previous_layer = np.zeros(n_units_previous)

        for j in range(n_units_L):
            bias_changes[L][j] = self.d_activation_funtion(Z[L][j]) * error[j]

        for k in range(n_units_previous):
            for j in range(n_units_L):
                error_previous_layer[k] += self.weights[k][j] * self.dactivation_funtion(Z[L][j]) * error[j]
                weight_changes[k][j] = A[k][j] * self.d_activation_funtion(Z[L][j]) * error[j]
        error_previous_layer /= j   
        self.backprop(A,Z,L-1,bias_changes, weight_changes, error_previous_layer)    

    def d_loss_function(self,y,y_hat):
        if (self.loss=='SSE'):
            return 2*(y_hat-y)
        else:
            print("delta loss function not implemented")
            sys.exit(0)

    def d_activation_funtion(self, x):
        if (self.act== 'sigmoid'):
            return logit(x) #Maybe not allowed?
        else:
            print("delta activation function not implemented")
            sys.exit(0)

    def backpropagation_batch(self,X,Y, learningRate):

        total_bias_changes = [np.zeros(biases_layer.shape) for biases_layer in self.biases]
        total_weight_changes = [np.zeros(weights_layer.shape) for weights_layer in self.weights]

        n_layers = len(self.weights)

        for x,y in zip(X,Y):
            A, Z = self.activations_forward_pass(x)

            bias_changes = [np.zeros(biases_layer.shape) for biases_layer in self.biases]
            weight_changes = [np.zeros(weights_layer.shape) for weights_layer in self.weights]

            errors = self.d_loss_function(A[-1], y)
            self.backprop(A,Z,n_layers,bias_changes,weight_changes, errors)

            total_biases_changes += bias_changes
            total_weight_changes += weight_changes
        

        self.biases += learningRate * total_bias_changes
        self.weights += learningRate * total_weight_changes
