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

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, = train_test_split(x_standard, y, train_size=0.8) 

#2. 모델 구성
model = Sequential()
model.add(Dense(200, activation='relu', input_shape=(4,)))
model.add(Dropout(0.2))
model.add(Dense(180, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(110, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(60, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='min') 

model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.25, verbose=1, callbacks=[early_stopping])

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
# loss :  0.03336896374821663
# y_test :  [0 0 2 1 0 2 2 0 2 1 2 1 0 1 2 2 2 2 2 0 2 1 1 2 0 2 1 0 0 0]
# y_predict :
#  [0.07372916 0.07422471 1.927014   1.0966015  0.07381046 1.1287266
#  1.9211011  0.0745827  1.8752325  1.094454   1.9337206  1.1227766
#  0.07416761 1.0864986  1.8819199  1.9499855  1.8770928  1.9423661
#  1.916281   0.07432961 1.8492433  1.0792978  1.1156299  1.900515
#  0.07515979 1.8811612  1.0797031  0.07354283 0.07377934 0.07421637]
# RMSE :  0.3382378178085266
# R2 :  0.955900046933227