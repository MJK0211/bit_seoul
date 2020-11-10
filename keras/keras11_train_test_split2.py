from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#1. 데이터
x = np.array(range(1,21))
y = np.array(range(31,51))

from sklearn.model_selection import train_test_split

x_train1, x_test, y_train1, y_test, = train_test_split(x, y, train_size=0.7) 
x_train2, x_val, y_train2, y_val = train_test_split(x_train1, y_train1, train_size=0.7)

print("x_train1 : ", x_train1)
print("x_train2 : ", x_train2)
print("x_test : ", x_test)
print("x_val : ", x_val)

# 결과값
# x_train1 :  [12  1 20 16 18  5 14  9  8 17  3  4 19 11]
# x_train2 :  [ 8 12 11  3 18  9 16 17 20]
# x_test :  [ 6 10  7 13  2 15]
# x_val :  [ 4  5 19 14  1]

print(x_test.shape)
