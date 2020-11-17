#Cifar10 LSTM

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LSTM
from tensorflow.keras.datasets import cifar10 #dataset인 cifar10추가
from tensorflow.keras.utils import to_categorical

#1. 데이터
#cifar10을 통해 비행기, 자동차, 새, 고양이, 사슴, 개, 개구리, 말, 배, 트럭 데이터 검출

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_predict = x_test[:10]
y_col = y_test[:10]

#1_1. 데이터 전처리 - OneHotEncoding

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(x_train.shape)

x_train = x_train.reshape(50000,32,96).astype('float32')/255. #CNN은 4차원이기 때문에 4차원으로 변환, astype -0 형변환
x_test = x_test.reshape(10000,32,96).astype('float32')/255.
x_predict = x_predict.reshape(10,32,96).astype('float32')/255

#2. 모델구성 
model = Sequential()
model.add(LSTM(5, activation='relu', input_shape=(32,96))) 
model.add(Dense(10, activation='softmax')) 
model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) #mse - 'mean_squared_error' 가능 
#다중분류에서는 categorical_crossentropy를 사용한다!

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
early_stopping = EarlyStopping(monitor='loss', patience=2, mode='min') 
#to_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True) 
model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping])

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", loss)
print("acc : ", acc)

y_pred = model.predict(x_predict)
y_pred = np.argmax(y_pred, axis=1)
print("y_col : ", y_col)
print("y_pred : ", y_pred)

# 결과값
# loss :  1.8047091960906982
# acc :  0.321399986743927
# y_col :  [[3]
#  [8]
#  [8]
#  [0]
#  [6]
#  [6]
#  [1]
#  [6]
#  [3]
#  [1]]
# y_pred :  [2 9 8 8 4 6 8 6 5 8]