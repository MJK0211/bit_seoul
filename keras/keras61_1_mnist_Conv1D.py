#36_mnist2_CNN 버전을 LSTM 코딩하기
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, MaxPooling1D, Flatten, LSTM, Conv1D

#1. 데이터
#mnist를 통해 손글씨 1~9까지의 데이터를 사용

x_train = np.load('./data/npy/mnist_x_train.npy') #(60000, 28, 28)
x_test = np.load('./data/npy/mnist_x_test.npy') #(10000, 28, 28)
y_train = np.load('./data/npy/mnist_y_train.npy') #(60000,)
y_test = np.load('./data/npy/mnist_y_test.npy') #(10000,)

x_predict = x_test[:10]
x_test = x_test[10:] #(9990,28,28)
y_real = y_test[:10] #(10,)
y_test = y_test[10:] #(9990,)

#1_1. 데이터 전처리 - OneHotEncoding
from tensorflow.keras.utils import to_categorical #keras
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape) #(60000, 10)
print(y_train[0]) #5 - [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.] - 0/1/2/3/4/5/6/7/8/9 - 5의 위치에 1이 표시됨

x_train = x_train.astype('float32')/255. 
x_test = x_test.astype('float32')/255.
x_predict = x_predict.astype('float32')/255

#2. 모델구성 
model = Sequential()
model.add(Conv1D(32, (2), padding='same', input_shape=(28,28))) 
model.add(Conv1D(64, (2))) 
model.add(Conv1D(128, (2))) 
model.add(Conv1D(64, (2))) 
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax')) 
model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) # mse - 'mean_squared_error' 가능

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='min')  
model.fit(x_train, y_train, epochs=30, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping])

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", loss)
print("acc : ", acc)

y_pred = model.predict(x_predict)
y_pred = np.argmax(y_pred, axis=1)
print("y_real : ", y_real)
print("y_pred : ", y_pred)

# 결과값
# loss :  0.13793812692165375
# acc :  0.9747747778892517
# y_real :  [7 2 1 0 4 1 4 9 5 9]
# y_pred :  [7 2 1 0 4 1 4 9 5 9]