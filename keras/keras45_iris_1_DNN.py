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

x_pred = x[:10] #(10,4)
y_real = y[:10] #(10,)
x = x[10:] #(140,4)
y = y[10:] #(140,)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x_minmax = scaler.transform(x) 
x_pred_minmax = scaler.transform(x_pred)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, = train_test_split(x_minmax, y, train_size=0.8) 

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

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
model.add(Dense(3, activation='softmax'))
model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='min') 

model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.25, verbose=1, callbacks=[early_stopping])

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("acc : ", acc)

y_predict = model.predict(x_pred_minmax)
y_predict = np.argmax(y_predict, axis=1)
#print(y_predict.shape) #(30, 1)

print("y_real : ", y_real)
print("y_predict : \n", y_predict)

# 결과값
# loss :  0.022439301013946533
# acc :  1.0
# y_real :  [0 0 0 0 0 0 0 0 0 0]
# y_predict :
#  [0 0 0 0 0 0 0 0 0 0]