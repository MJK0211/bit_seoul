#36_mnist2_CNN 버전을 LSTM 코딩하기
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, MaxPooling1D, Flatten, LSTM, Conv1D

#1. 데이터

x_train = np.load('./data/npy/diabetes_x_train.npy') 
x_test = np.load('./data/npy/diabetes_x_test.npy') 
y_train = np.load('./data/npy/diabetes_y_train.npy') 
y_test = np.load('./data/npy/diabetes_y_test.npy')

x_predict = x_test[:10]
x_test = x_test[10:] 
y_real = y_test[:10] 
y_test = y_test[10:] 

print(x_train.shape) #(353, 10)
print(x_test.shape) #(79, 10)
print(y_train.shape) #(353,)
print(y_test.shape) #(79,)
print(y_real.shape) #(10,)


#1.1 데이터 전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train_minmax = scaler.transform(x_train)
x_test_minmax = scaler.transform(x_test)
x_predict_minmax = scaler.transform(x_predict)

x_train_minmax = x_train_minmax.reshape(353,10,1)
x_test_minmax = x_test_minmax.reshape(79,10,1)
x_pred_minmax = x_predict_minmax.reshape(10,10,1)

#2. 모델구성 
model = Sequential()
model.add(Conv1D(32, (2), padding='same', input_shape=(10,1))) 
model.add(Conv1D(64, (1))) 
model.add(Conv1D(128, (1))) 
model.add(Conv1D(64, (1))) 
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1)) 
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
early_stopping = EarlyStopping(monitor='loss', patience=50, mode='min')  
model.fit(x_train_minmax, y_train, epochs=500, batch_size=12, verbose=1, validation_split=0.2, callbacks=[early_stopping])

#4. 평가, 예측
loss = model.evaluate(x_test_minmax, y_test, batch_size=12)
print("loss : ", loss)

y_pred = model.predict(x_pred_minmax)
y_pred = np.argmax(y_pred, axis=1)
print("y_real : ", y_real.reshape(10,))
print("y_pred : ", y_pred)

# 결과값
# loss :  3707.8310546875
# y_real :  [ 60.  59. 210.  84. 243. 163. 308. 182.  94.  51.]
# y_pred :  [0 0 0 0 0 0 0 0 0 0]