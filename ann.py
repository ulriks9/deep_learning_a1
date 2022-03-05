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

    #add a hidden layer, the last hidden layer added serves as output layer
    def add_hidden(self, n_units):
        self.weights.append(np.random.uniform(low=-0.7, high=0.7, size=(n_units, self.n_units[-1])))
        self.biases.append(np.random.uniform(low=-0.7, high=0.7, size=n_units))
        self.n_units.append(n_units)

    #calculates a forward pass, given an input and returns the activations of the output layer
    def forward_pass(self, x):
        for w_matrix, b_matrix in zip(self.weights, self.biases):
            x = np.dot(x, np.transpose(w_matrix)) + b_matrix
            x = self.activationFunction(x)
        return x

    #predict a class given an input
    def predict(self, x):
        activation = self.forward_pass(x)[0]
        if activation > 0.5:
            return 1
        return 0

    #performs a forward pass and returns activations before and after the activation function is applied
    #this is used for the backpropagation algorithm
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
    

    #a recursive function that propagates the error backwards over the layers and calculates the desired weight and
    #bias changes. The desired changes are written into the inputs "bias_changes" and "weight_changes"
    def backprop(self,A,Z,L,bias_changes,weight_changes,error):
        #Reached end of the network
        if L == -1:
            return

        n_units_L = len(self.biases[L])
        n_units_previous = len(self.weights[L][0])

        error_previous_layer = np.zeros(n_units_previous)

        for j in range(n_units_L):
            #calculated bias changes, proportional to derivative of activation function
            bias_changes[L][j] = self.d_activation_function(Z[L][j]) * error[j]

        for k in range(n_units_previous):
            for j in range(n_units_L):
                #calculated propagated errors, proportional to derivative of activation functionand and weight
                error_previous_layer[k] += self.weights[L][j][k] * self.d_activation_function(Z[L][j]) * error[j]
                #calculated weight changes, proportional to derivative of activation functionand and activation of connected unit
                weight_changes[L][j][k] = A[L][k] * self.d_activation_function(Z[L][j]) * error[j]
        #recursively call for previous layer
        self.backprop(A,Z,L-1,bias_changes, weight_changes, error_previous_layer)    

    #returns derivative of loss function
    def d_loss_function(self,y,y_hat):
        if (self.loss=='SSE'):
            return 2*(y_hat-y)
        else:
            print("delta loss function not implemented")
            sys.exit(0)

    #returns derivative of activation function
    def d_activation_function(self, x):
        if (self.act== 'sigmoid'):
            y = 1 / (1+ np.exp(-x))
            return y * (1-y) 
        elif self.act == 'tanh':
            return 1 - math.tanh(x) * math.tanh(x)
        else:
            print("derivative of activation function not implemented")
            sys.exit(0)
    
    #calculates activation of unit given the sum of raw inputs
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

    #returns square sum of error and correctness of output, given the activations of output layer and desired output
    def evaluate(self,x,y):
        sse = 0
        if (self.loss=='SSE'):
            sse += (x-y)**2
        else:
            print("loss function not implemented")
            sys.exit(0)
        
        if (x > 0.5 and y == 1 ) or (x <= 0.5 and y == 0):
            return sse, 1
        return sse, 0

    #backpropagation for one batch, returns the total changes based on batch
    def backpropagation_batch(self,X,Y,verbose):

        total_bias_changes = [np.zeros(biases_layer.shape) for biases_layer in self.biases]
        total_weight_changes = [np.zeros(weights_layer.shape) for weights_layer in self.weights]

        n_layers = len(self.weights)

        square_errors = 0
        accuracy = 0

        for x,y in zip(X,Y):
            A, Z = self.activations_forward_pass(x)
            se, correct = self.evaluate(A[-1][0],y)
            square_errors += se
            accuracy += correct

            bias_changes = [np.zeros(biases_layer.shape) for biases_layer in self.biases]
            weight_changes = [np.zeros(weights_layer.shape) for weights_layer in self.weights]

            errors = self.d_loss_function(A[-1], y)
            #call the backprop algorihm for one input output pair
            self.backprop(A,Z,n_layers-1,bias_changes,weight_changes, errors)

            #sum up the total changes
            total_bias_changes = np.add(bias_changes, total_bias_changes)
            total_weight_changes = np.add(weight_changes, total_weight_changes)

        if verbose:
            print("accuracy = {0}%, mean square error = {1}".format(round((accuracy *100) / len(X),2), round(square_errors / len(X),3 )))

        return total_bias_changes, total_weight_changes

    #train the network using a specified number of epochs, momentum and learning rate.
    #the learning rate is a range and changed is changed over the epochs in an logarithmic decrease
    def train(self,data,n_epochs = 100, batch_size = 40, momentum = 0.8, learning_rate = [1,0.1], verbose = False):

        previous_changes_biases = None
        previous_changes_weights = None

        epoch =0
        for rate in np.linspace(learning_rate[0]**0.5, learning_rate[1]**0.5,num=n_epochs)**2:
            epoch +=1
            #samples an batch, batch is gatherd randomly from training data with replacement
            data_epoch = [data[x] for x in sample(range(0,len(data)),batch_size)]
            X = [pair[0] for pair in data_epoch]
            Y = [pair[1] for pair in data_epoch]
            if verbose:
                print("Epoch {0} with lr = {1}: ".format(epoch,rate),end = "")
            bias_changes, weight_changes = self.backpropagation_batch(X,Y,verbose)

            previous_biases = self.biases
            previous_weights = self.weights

            self.biases =  np.add(self.biases,  bias_changes* rate)
            self.weights = np.add(self.weights,  weight_changes* rate)

            if (previous_changes_weights!=None):
                self.biases =  np.add(self.biases,  previous_changes_biases *  momentum)
                self.weights =  np.add(self.weights,  previous_changes_weights *  momentum)

            #remember previous changes for momentum
            previous_changes_weights = np.subtract(self.weights,previous_weights)
            previous_changes_biases = np.subtract(self.biases,previous_biases)

    #returns accuracy given a set of input-output pairs
    def  accuracy(self,X,Y):
        accuracy = 0

        for x,y in zip(X,Y):
            if y == self.predict(x):
                accuracy +=1

        return (accuracy / len(X))
