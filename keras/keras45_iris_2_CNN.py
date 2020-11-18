#load_iris - CNN

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.datasets import load_iris #load_iris 꽃 확인

#1. 데이터
# Attribute Information (in order):
#     0    - sepal length in cm    꽃받침 길이(Sepal Length)
#     1    - sepal width in cm     꽃받침 폭(Sepal Width)
#     2    - petal length in cm    꽃잎 길이(Petal Length)
#     3    - petal width in cm     꽃잎 폭(Petal Width)
#     4     - class:               setosa, versicolor, virginica의 세가지 붓꽃 종(species)
                # - Iris-Setosa 
                # - Iris-Versicolour
                # - Iris-Virginica

dataset = load_iris()
x = dataset.data #(150,4)
y = dataset.target #(150,)
#x의 데이터로 세가지 붓꽃 종 중 하나를 찾는 데이터셋이다

print(x.shape)
print(y.shape)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
scaler.fit(x)
x_standard = scaler.transform(x) 
x_standard = x_standard.reshape(150,2,2,1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, = train_test_split(x_standard, y, train_size=0.8) 

#2. 모델 구성
model = Sequential()
model.add(Conv2D(10, (5,2), padding='same', input_shape=(2,2,1)))
model.add(Dropout(0.2))
model.add(Conv2D(20, (2,2), padding='same')) 
model.add(Dropout(0.2))
model.add(Conv2D(30, (2,2), padding='same'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=2)) 
model.add(Dropout(0.2))
model.add(Flatten()) 
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='softmax')) 
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='min') 

model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.25, verbose=1, callbacks=[early_stopping])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

#print(y_predict.shape) #(30, 1)

print("y_test : ", y_test)
print("y_predict : \n", y_predict.reshape(30,))

from sklearn.metrics import mean_squared_error, r2_score

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

# 결과값
# loss :  0.6666666865348816
# y_test :  [0 1 2 0 1 0 0 2 2 0 1 1 2 2 0 0 0 2 1 1 1 2 0 1 0 1 0 0 1 0]
# y_predict :
#  [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
#  1. 1. 1. 1. 1. 1.]
# RMSE :  0.816496580927726
# R2 :  -0.06382978723404231