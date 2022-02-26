from typing import List
from scipy.special import expit
import numpy as np
import sys
import math
import warnings
from random import sample
from sklearn.utils import shuffle
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

class ANN:

    def __init__(self, act='sigmoid', loss='SSE', n_features=2):
        self.weights = []
        self.biases = []
        self.act = act
        self.n_units = []
        self.n_units.append(n_features)
        self.loss = loss

    def add_hidden(self, n_units):
        self.weights.append(np.random.uniform(low=-0.7, high=0.7, size=(n_units, self.n_units[-1])))
        self.biases.append(np.random.uniform(low=-0.7, high=0.7, size=n_units))
        self.n_units.append(n_units)

    def forward_pass(self, x):
        for w_matrix, b_matrix in zip(self.weights, self.biases):
            x = np.dot(x, np.transpose(w_matrix)) + b_matrix
            x = self.activationFunction(x)
        return x

    def predict(self, x):
        return np.argmax(self.forward_pass(x))

    def activations_forward_pass(self,x):
        A = []
        Z = []
        A.append(x)
        for w_matrix, b_matrix in zip(self.weights, self.biases):
            x = np.dot(x, np.transpose(w_matrix)) + b_matrix
            Z.append(x)
            x = self.activationFunction(x)
            A.append(x)
        return A,Z
    

    def backprop(self,A,Z,L,bias_changes,weight_changes,error):
        if L == -1:
            return

        n_units_L = len(self.biases[L])
        n_units_previous = len(self.weights[L][0])

        error_previous_layer = np.zeros(n_units_previous)

        for j in range(n_units_L):
            bias_changes[L][j] = self.d_activation_function(Z[L][j]) * error[j]

        for k in range(n_units_previous):
            for j in range(n_units_L):
                error_previous_layer[k] += self.weights[L][j][k] * self.d_activation_function(Z[L][j]) * error[j]
                weight_changes[L][j][k] = A[L][k] * self.d_activation_function(Z[L][j]) * error[j]
        #error_previous_layer = error_previous_layer / n_units_L  
        self.backprop(A,Z,L-1,bias_changes, weight_changes, error_previous_layer)    

    def d_loss_function(self,y,y_hat):
        if (self.loss=='SSE'):
            return 2*(y_hat-y)
        else:
            print("delta loss function not implemented")
            sys.exit(0)

    def d_activation_function(self, x):
        if (self.act== 'sigmoid'):
            y = 1 / (1+ np.exp(-x))
            return y * (1-y) 
        elif self.act == 'tanh':
            return 1 - math.tanh(x) * math.tanh(x)
        else:
            print("derivative of activation function not implemented")
            sys.exit(0)
    
    def activationFunction(self,x):
        if self.act == 'sigmoid':
            return  expit(x)
        elif self.act == 'tanh':
            try:
                return math.tanh(x)
            except TypeError:
                return [math.tanh(x_i) for x_i in x]
        else:
            print("activation function '{0}' is not implemented".format(self.act))
            sys.exit(0)

    def evaluate(self,x,y):
        sse = 0
        if (self.loss=='SSE'):
            for v_x,v_y in zip(x,y):
                sse += (v_x-v_y)**2
        else:
            print("loss function not implemented")
            sys.exit(0)
        correct = np.argmax(x) == np.argmax(y)
        return sse,correct


    def backpropagation_batch(self,X,Y,verbose):

        total_bias_changes = [np.zeros(biases_layer.shape) for biases_layer in self.biases]
        total_weight_changes = [np.zeros(weights_layer.shape) for weights_layer in self.weights]

        n_layers = len(self.weights)

        square_errors = 0
        accuracy = 0

        for x,y in zip(X,Y):
            A, Z = self.activations_forward_pass(x)
            se, correct = self.evaluate(A[-1],y)
            square_errors += se
            accuracy += correct

            bias_changes = [np.zeros(biases_layer.shape) for biases_layer in self.biases]
            weight_changes = [np.zeros(weights_layer.shape) for weights_layer in self.weights]

            errors = self.d_loss_function(A[-1], y)
            self.backprop(A,Z,n_layers-1,bias_changes,weight_changes, errors)

            total_bias_changes = np.add(bias_changes, total_bias_changes)
            total_weight_changes = np.add(weight_changes, total_weight_changes)

        if verbose:
            print("accuracy = {0}%, mean square error = {1}".format((accuracy *100) / len(X), square_errors / len(X) ))

        return total_bias_changes / len(X), total_weight_changes / len(X)
        
        
    def train(self,data,n_epochs = 100, batch_size = 500, momentum = 0.8, learing_rate = 0.1, verbose = True):

        previous_biases = None
        previous_weights = None

        for epoch in range(n_epochs):
            data_epoch = [data[x] for x in sample(range(0,len(data)),batch_size)]
            X = [pair[0] for pair in data_epoch]
            Y_raw = [pair[1] for pair in data_epoch]
            Y = [np.zeros(len(self.biases[-1])) for y in Y_raw]
            for i,y_class in enumerate(Y_raw):
                Y[i][y_class] = 1
            
            print("Epoch {0}: ".format(epoch),end = "")
            bias_changes, weight_changes = self.backpropagation_batch(X,Y,verbose)

            self.biases =  np.add(self.biases,  bias_changes * learing_rate)
            self.weights = np.add(self.weights,  weight_changes * learing_rate)

            if (previous_biases!=None):
                self.biases =  np.add(self.biases,  np.subtract(self.biases,previous_biases) *  momentum)
                self.weights =  np.add(self.weights,  np.subtract(self.weights,previous_weights) *  momentum)

            previous_biases = self.biases
            previous_weights = self.weights



            