#Cifar10 show plt

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.datasets import cifar10 #dataset인 cifar10추가
from tensorflow.keras.utils import to_categorical


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#print(x_train.shape, x_test.shape) #(50000, 32, 32, 3) (10000, 32, 32, 3) train 50000개, test 10000개
#print(y_train.shape, y_test.shape) #(50000, 1) (10000, 1) train 60000개, test 10000개

print(y_train[350])

# 0 비행기
# 1 자동차
# 2 새
# 3 고양이
# 4 사슴
# 5 개
# 6 개구리
# 7 말
# 8 배
# 9 트럭

plt.imshow(x_train[350])
plt.show()
