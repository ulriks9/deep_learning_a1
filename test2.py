from re import X

from sklearn.utils import shuffle
from torch import le
from data import *
from ann import *
from random import shuffle
from visualizeData import *

def print_accuracy(X,Y):
    accuracy = 0
    class1predictions = 0

    for x,y in zip(X,Y):
        pred = model.predict(x)
        if pred == y:
            accuracy +=1
        if pred == 0:
            class1predictions +=1
    print("Accuracy {0}, predictingclass 1 = {1}, predicting class 2 = {2}".format(accuracy/ len(X), class1predictions, len(X) - class1predictions))

data = create_data(N = 1000)
model = ANN(act = 'sigmoid',n_features=2)
model.add_hidden(12)
model.add_hidden(2)


lr = [3,0.7,0.2,0.03,0.01,0.002,0.0005]
for rate in lr:
    model.train(data,n_epochs=100,batch_size=500,momentum=0.7,learing_rate=rate,verbose=True)

new_data = create_data(N = 1000)
X = [pair[0] for pair in new_data]
Y = [pair[1] for pair in new_data]
print_accuracy(X,Y)

visualize_data(new_data)




