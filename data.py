from random import random
from sklearn.datasets import make_gaussian_quantiles

# Creates artificial binary data
def create_data(N=500, n_f=2, n_c=2):
    gq = make_gaussian_quantiles(
        mean=None,
        cov=0.7,
        n_samples=N,
        n_features=n_f,
        n_classes=n_c,
        shuffle=True,
        random_state=None)

    return gq
