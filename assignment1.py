from distutils.command.build import build
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Sequential
from data import *

def main():
    dataset = create_data(N=2500)
    x = np.array([x for x, _ in dataset])
    y = np.array([y for _, y in dataset])
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
    
    # Visualize data
    print("INFO: Building data visualization plot...")
    class1 = [x[0] for x in dataset if x[1] == 0]
    class2 = [x[0] for x in dataset if x[1] == 1]

    fig, ax = plt.subplots()
    ax.scatter([x[0] for x in class1], [x[1] for x in class1], label='class1')
    ax.scatter([x[0] for x in class2], [x[1] for x in class2], label='class2')
    ax.legend()
    ax.set_title('Data')
    plt.show()

    # Train and evaluate model
    test_model((x_train, x_test, y_train, y_test), structure=(2, 100, 100, 1), epochs=200)

# Returns compiled Keras model with specified structure
def build_model(structure, act='relu', loss='mean_squared_error', optimizer='sgd'):
    model = Sequential()

    # Add first hidden layer with input dimensions
    model.add(layers.Dense(structure[1], input_dim=structure[0], activation=act))

    for n in structure[2:]:
        model.add(layers.Dense(n, activation=act))
    
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    return model

# Displays average performance over specified amount of runs
def test_model(data, structure, runs=10, epochs=100):
    results = np.zeros((10,1))
    x_train, x_test, y_train, y_test = data

    print("INFO: Testing model over {} runs...\n".format(runs))

    for i in range(runs):
        model = build_model(structure=structure)
        model.fit(x_train, y_train, epochs=epochs, verbose=0)
        results[i] = model.evaluate(x_test, y_test, verbose=0)[1]
        
        print("INFO: Accuracy of run {}: {}".format(i + 1, np.round(results[i], decimals=3)))

    std = np.round(np.std(results), decimals=3)
    avg = np.round(np.average(results), decimals=3)

    print("INFO: Average accuracy over {} runs: {} +- {} std. deviation".format(runs, avg, std))

def evaluate(pred, y):
    accuracy = accuracy_score(y, pred)
    print("INFO: Total accuracy: {}".format(accuracy))


main()