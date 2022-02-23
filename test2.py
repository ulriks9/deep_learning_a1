from data import *
from ann import *

data = create_data()
model = ANN()
model.add_hidden(5)
model.add_hidden(2)
print(model.predict(data[0][0]))

model.backpropagation_batch(data[:][0],data[:][1] , 0.005)

print(model.predict(data[0][0]))
