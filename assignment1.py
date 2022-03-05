from calendar import leapdays
from distutils.command.build import build
from operator import le
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Sequential
from data import *
from ann import *
from visualizeData import *

def main():
    dataset = create_data(N=1000)
    x = np.array([x for x, _ in dataset])
    y = np.array([y for _, y in dataset])
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
    
    # Visualize data
    print("INFO: Building data visualization plot...")
    visualize_data(dataset)

    # Train and evaluate model
    test_model((x_train, x_test, y_train, y_test),runs = 10, structure=(2, 12, 1),batch_size = 20, epochs=500)
    test_handmade_model((x_train, x_test, y_train, y_test),runs = 10, structure=(2, 12, 1),learning_rates = (0.1,0.01), epochs = 500, batch_size = 20)

# Returns compiled Keras model with specified structure
def build_model(structure, act='relu', loss='mean_squared_error', optimizer='sgd'):
    model = Sequential()

    # Add first hidden layer with input dimensions
    model.add(layers.Dense(structure[1], input_dim=structure[0], activation=act))

    for n in structure[2:]:
        model.add(layers.Dense(n, activation=act))
    
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    return model

# Returns a new handcrafted model with specified structure
def build_handcrafted_model(structure, act = 'sigmoid' ):
    model = ANN(act = act,n_features=structure[0])
    #add hidden layers
    for x in structure[1:]:
        model.add_hidden(x)
    return model

# Displays average performance over specified amount of runs
def test_model(data, structure, runs=10,batch_size = 20, epochs=100):
    results = np.zeros((runs,1))
    x_train, x_test, y_train, y_test = data

    print("INFO: Testing model over {} runs...\n".format(runs))

    for i in range(runs):
        model = build_model(structure=structure)
        model.fit(x_train, y_train,batch_size = batch_size, epochs=epochs, verbose=0)
        results[i] = model.evaluate(x_test, y_test, verbose=0)[1]
        
        print("INFO: Accuracy of keras model run {}: {}".format(i + 1, np.round(results[i], decimals=3)))

    std = np.round(np.std(results), decimals=3)
    avg = np.round(np.average(results), decimals=3)

    print("INFO: Average accuracy of keras model over {} runs: {} +- {} std. deviation".format(runs, avg, std))

# Displays average performance over specified amount of runs
def test_handmade_model(data, structure, runs=10, learning_rates = (10,0.1), epochs = 500, batch_size = 20):
    results = np.zeros((runs,1))
    x_train, x_test, y_train, y_test = data
    data_train = [[x,y] for x,y in zip(x_train,y_train)]

    print("INFO: Testing handcrafted model over {} runs...\n".format(runs))

    for i in range(runs):
        model = build_handcrafted_model(structure=structure)
        
        model.train(data_train,n_epochs=epochs,batch_size=batch_size,learning_rate=learning_rates,verbose=False)

        results[i] = model.accuracy(x_test,y_test)
        
        print("INFO: Accuracy of handcrafted model run {}: {}".format(i + 1, np.round(results[i], decimals=3)))

    std = np.round(np.std(results), decimals=3)
    avg = np.round(np.average(results), decimals=3)

    print("INFO: Average accuracy of handcrafted model over {} runs: {} +- {} std. deviation".format(runs, avg, std))

main()