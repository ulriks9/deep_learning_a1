from data import *
from ann import *

data = create_data()
model = ANN()

model.add_hidden(10)
model.add_hidden(2)

output = model.predict(data[0][0])


print(output)