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
                # 0 - Iris-Setosa     
                # 1 - Iris-Versicolour
                # 2 - Iris-Virginica

dataset = load_iris()
x = dataset.data #(150,4)
y = dataset.target #(150,)
#x의 데이터로 세가지 붓꽃 종 중 하나를 찾는 데이터셋이다

print(x.shape) #(150, 4)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8) 
x_pred = x_test[:10]
x_test = x_test[10:]
y_real = y_test[:10]
y_test = y_test[10:]

print(x_test.shape) #(20, 4)
print(y_test.shape) #(20,)
print(x_train.shape) #(120,4)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train_minmax = scaler.transform(x_train) 
x_test_minmax = scaler.transform(x_test)
x_pred_minmax = scaler.transform(x_pred)

x_train_minmax = x_train_minmax.reshape(120,4,1,1)
x_test_minmax = x_test_minmax.reshape(20,4,1,1)
x_pred_minmax = x_pred.reshape(10,4,1,1)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(32, (2, 2), padding='same', activation='relu', input_shape=(4, 1, 1))) #padding 주의!
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(256, (2, 2), padding='same', activation='relu'))
model.add(Dropout(0.2))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax')) 
model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping(monitor='val_loss', patience=100, mode='min') 

model.fit(x_train_minmax, y_train, epochs=1000, batch_size=32, validation_split=0.25, verbose=1, callbacks=[early_stopping])

#4. 평가, 예측
loss, acc = model.evaluate(x_test_minmax, y_test, batch_size=32)
print("loss : ", loss)
print("acc : ", acc)

y_predict = model.predict(x_pred_minmax)
y_predict = np.argmax(y_predict, axis=1)

print("y_real : ", y_real)
print("y_predict : \n", y_predict)

# 결과값
# loss :  0.16634956002235413
# acc :  0.949999988079071
# y_real :  [2 1 1 2 1 2 2 2 1 0]
# y_predict :
#  [2 1 1 2 1 2 2 2 2 1]