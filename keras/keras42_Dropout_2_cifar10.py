#Dropout - Cifar10_CNN

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
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

x_train = x_train.astype('float32')/255. 
x_test = x_test.astype('float32')/255.
x_predict = x_predict.astype('float32')/255

#2. 모델구성 
model = Sequential()
model.add(Conv2D(60, (2,2), padding='same', input_shape=(32,32,3))) 
model.add(Dropout(0.2))
model.add(Conv2D(50, (2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(40, (3,3), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(30, (2,2), strides=2))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=2)) 
model.add(Dropout(0.2))
model.add(Flatten()) 
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax')) 
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
# loss :  1.284420132637024
# acc :  0.5924999713897705
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
# y_pred :  [3 8 8 8 6 6 1 6 3 1]