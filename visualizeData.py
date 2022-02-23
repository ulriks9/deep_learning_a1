import matplotlib.pyplot as plt
from data import *

data  = create_data()

print(data[0][0])
print(len(data[0]))
print(len(data[1]))

class1Data = [x[0] for x in data if x[1] == 0]
class2Data = [x[0] for x in data if x[1] == 1]

plt.scatter([x[0] for x in class1Data],[x[1] for x in class1Data], label = "class1")
plt.scatter([x[0] for x in class2Data],[x[1] for x in class2Data], label = "class2")
plt.legend()
plt.show()

