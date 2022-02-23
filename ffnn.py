from turtle import backward, forward
import numpy as np
from utils import *

class ANN:

    def __init__(self, struct, l_R=0.001):
        self.struct = struct
        self.n_layers = len(struct)
        self.l_R = l_R

        # Generates weights and biases based on np's randn
        self.B = [np.random.randn(n, 1) for n in struct[1:]]
        self.W = [np.random.randn(n, n_next) for n, n_next in zip(struct[:-1], struct[1:])]

    def predict(self, x):
        for b, W in zip(self.B, self.W):
            x = sigmoid(np.dot(W.T, x) + b.T[0])
            
        return np.argmax(x)

    # Computes a forward pass and returns activations and pre-activations
    def forward(self, x):
        A = []
        Z = [] 

        for b, W in zip(self.B, self.W):
            z = np.dot(W.T, x)
            print("Z")
            print(z)
            z += b.T[0]
            x = sigmoid(z)
            Z.append(z)
            A.append(x)
    
        return A, Z
    
    # Calulates backward pass and returns the errors w.r.t. weights and biases for each layer
    def backward(self, x, y):
        # Performs forward pass
        A, Z = self.forward(x)

        # Index of last hidden layer
        i = self.n_layers - 2

        # Errors w.r.t. each layer
        dE_dB = [np.zeros(b.shape) for b in self.B]
        dE_dW = [np.zeros(w.shape) for w in self.W]

        for L in range(i, -1, -1):
            # Last layer needs to compute error
            if L != i:
                # Backpropagation of error from L + 1
                delta = d_sigmoid(Z[L]) * (self.W[L + 1] @ delta)
            else:
                # Derivative of MSE w.r.t. weights in layer (Z is a function of W)
                delta = (A[L] - y) * d_sigmoid(Z[L])

            dE_dB[L] = delta
            # First layer needs to use input for calculating error contribution
            if L != 0:
                dE_dW[L] = A[L - 1] * delta.T
            else:
                dE_dW[L] = x * delta.T

        return dE_dB, dE_dW

    # Performs Batch Gradient Descent on given batch
    def BGD(self, batch):
        # Errors w.r.t. each layer#ag
        dE_dB_total = [np.zeros(b.shape) for b in self.B]
        dE_dW_total = [np.zeros(w.shape) for w in self.W]

        # Number of examples in batch
        P = len(batch)

        for x, y in batch:
            dE_dB, dE_dW = self.backward(x, y)

            # Sums to the total amount to update weights and biases with
            dE_dB_total = [dE_dB_old + dE_dB_new for dE_dB_old, dE_dB_new in zip(dE_dB_total, dE_dB)]
            dE_dW_total = [dE_dW_old + dE_dW_new for dE_dW_old, dE_dW_new in zip(dE_dW_total, dE_dW)]
        
        # Updates weights and biases
        self.B = [b - (self.l_R / P) * dB for b, dB in zip(self.B, dE_dB_total)]
        self.W = [w - (self.l_R / P) * dW for w, dW in zip(self.W, dE_dW_total)]

