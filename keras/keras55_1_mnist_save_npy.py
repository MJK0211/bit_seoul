import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.datasets import mnist #dataset인 mnist추가

#1. 데이터
#mnist를 통해 손글씨 1~9까지의 데이터를 사용
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, x_test.shape) #(60000,28,28), (10000,28,28)
# print(y_train.shape, y_test.shape) #(60000,), (10000,)

np.save('./data/npy/mnist_x_train.npy', arr=x_train)
np.save('./data/npy/mnist_x_test.npy', arr=x_test)
np.save('./data/npy/mnist_y_train.npy', arr=y_train)
np.save('./data/npy/mnist_y_test.npy', arr=y_test)
