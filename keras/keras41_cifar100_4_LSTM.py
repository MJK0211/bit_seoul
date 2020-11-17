#Cifar100 LSTM

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LSTM
from tensorflow.keras.datasets import cifar100 #dataset인 cifar100추가
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_predict = x_test[:10]
y_col = y_test[:10]

#1_1. 데이터 전처리 - OneHotEncoding

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(50000,32,96).astype('float32')/255. #CNN은 4차원이기 때문에 4차원으로 변환, astype -0 형변환
x_test = x_test.reshape(10000,32,96).astype('float32')/255.
x_predict = x_predict.reshape(10,32,96).astype('float32')/255

#2. 모델구성 
model = Sequential()
model.add(LSTM(5, activation='relu', input_shape=(32,96))) 
model.add(Dense(100, activation='softmax')) 
model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) 

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='min') 
#to_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True) 
model.fit(x_train, y_train, epochs=30, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping])

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", loss)
print("acc : ", acc)

y_pred = model.predict(x_predict)
y_pred = np.argmax(y_pred, axis=1)
print("y_col : ", y_col)
print("y_pred : ", y_pred)

# 결과값
# loss :  3.843097686767578
# acc :  0.10329999774694443
# y_col :  [[49]
#  [33]
#  [72]
#  [51]
#  [71]
#  [92]
#  [15]
#  [14]
#  [23]
#  [ 0]]
# y_pred :  [39 97 32  5 54 14 27 75 71 41]