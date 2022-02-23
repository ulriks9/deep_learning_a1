from random import random
from sklearn.datasets import make_gaussian_quantiles

# Creates artificial binary data
def create_data(N=500, n_f=2, n_c=2):
    data = []

    gq = make_gaussian_quantiles(
        mean=None,
        cov=0.7,
        n_samples=N,
        n_features=n_f,
        n_classes=n_c,
        shuffle=True,
        random_state=None)

    for i in range(len(gq)):
        data.append((gq[0][i], gq[1][i]))

    return data
