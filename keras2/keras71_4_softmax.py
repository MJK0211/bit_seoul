import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

x = np.arange(1,5)
y = softmax(x)

print(y)

ratio = y
labels = y

plt.pie(ratio, labels=labels, shadow=True, startangle=90)
plt.grid()
plt.show()
