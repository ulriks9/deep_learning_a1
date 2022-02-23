from re import X

from sklearn.utils import shuffle
from data import *
from ann import *
from random import shuffle

def print_accuracy(X,Y,epoch):
    accuracy = 0
    class1predictions = 0

    for x,y in zip(X,Y):
        pred = model.predict(x)
        if pred == y:
            accuracy +=1
        if pred == 0:
            class1predictions +=1
    print("Epoch {0}: Accuracy {1}, predictingclass 1 = {2}, predicting class 2 = {3}".format(epoch,accuracy/ len(X), class1predictions, len(X) - class1predictions))

data = create_data(N = 500)
model = ANN(n_features=2)
model.add_hidden(10)
model.add_hidden(2)


#X = [[1,1],[0,1],[1,0],[0,0]]
#Y = [[1],[0],[0],[1]]

for i in range(200):
    shuffle(data)
    X = [pair[0] for pair in data]
    Y = [pair[1] for pair in data]
    model.backpropagation_batch(X,Y, 10)
    print_accuracy(X,Y,i)




