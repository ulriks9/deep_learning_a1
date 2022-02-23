import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def d_sigmoid(x):
     sigmoid(x) * (1 - sigmoid(x))