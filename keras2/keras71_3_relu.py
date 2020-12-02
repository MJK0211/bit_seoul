import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

x = np.arange(-5, 5, 0.1) #0이하는 0, 이상은 x값
y = relu(x)

print(y)

plt.plot(x,y)
plt.grid()
plt.show()

#relu 친구들 찾기
#elu, selu, ???elu 검색해서 밑에 주석정리하기