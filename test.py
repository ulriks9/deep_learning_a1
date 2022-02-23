from data import *
#from ann import *
from ffnn import *

data = create_data()
model = ANN((2, 3, 2))


output = model.predict(data[0][0])


print(output)