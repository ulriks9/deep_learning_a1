

import matplotlib
import matplotlib.pyplot as plt


import sklearn.datasets


def load_data():
    N = 500
    gq = sklearn.datasets.make_gaussian_quantiles(
        mean=None ,
        cov =0.7 ,
        n_samples =N,
        n_features =2,
        n_classes =2,
        shuffle=True ,
        random_state =None)
    return gq

data  = load_data()

print(data[0][0])
print(len(data[0]))
print(len(data[1]))

class1Data = [x for x,y in zip(data[0],data[1]) if y == 0]
class2Data = [x for x,y in zip(data[0],data[1]) if y == 1]

plt.scatter([x[0] for x in class1Data],[x[1] for x in class1Data], label = "class1")
plt.scatter([x[0] for x in class2Data],[x[1] for x in class2Data], label = "class2")
plt.legend()
plt.show()


