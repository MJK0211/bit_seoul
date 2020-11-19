#36_mnist2_CNN 버전을 LSTM 코딩하기
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, MaxPooling1D, Flatten, LSTM, Conv1D

#1. 데이터
#mnist를 통해 손글씨 1~9까지의 데이터를 사용

x_train = np.load('./data/npy/cancer_x_train.npy') 
x_test = np.load('./data/npy/cancer_x_test.npy') 
y_train = np.load('./data/npy/cancer_y_train.npy') 
y_test = np.load('./data/npy/cancer_y_test.npy')

x_predict = x_test[:10]
x_test = x_test[10:] 
y_real = y_test[:10] 
y_test = y_test[10:] 

print(x_train.shape) #(455, 30)
print(x_test.shape) #(104, 30)
print(y_train.shape) #(455,)
print(y_test.shape) #(104,)
print(y_real.shape) #(10,)


#1_1. 데이터 전처리
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train_minmax = scaler.transform(x_train) 
x_test_minmax = scaler.transform(x_test)
x_pred_minmax = scaler.transform(x_predict)

x_train_minmax = x_train_minmax.reshape(455,30,1)
x_test_minmax = x_test_minmax.reshape(104,30,1)
x_pred_minmax = x_pred_minmax.reshape(10,30,1)

#2. 모델구성 
model = Sequential()
model.add(Conv1D(32, (2), padding='same', input_shape=(30,1))) 
model.add(Conv1D(64, (2))) 
model.add(Conv1D(128, (2))) 
model.add(Conv1D(64, (2))) 
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid')) 
model.summary()

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) # mse - 'mean_squared_error' 가능

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
early_stopping = EarlyStopping(monitor='loss', patience=50, mode='min')  
model.fit(x_train_minmax, y_train, epochs=500, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping])

#4. 평가, 예측
loss, acc = model.evaluate(x_test_minmax, y_test, batch_size=32)
print("loss : ", loss)
print("acc : ", acc)

y_predict = model.predict(x_pred_minmax)
y_predict = np.round(y_predict.reshape(10,),0)
print("y_real : ", y_real)
print("y_pred : ", y_predict)

# 결과값
# loss :  0.0028218200895935297
# acc :  1.0
# y_real :  [1 1 1 0 0 1 1 1 0 1]
# y_pred :  [1. 1. 1. 0. 0. 1. 1. 1. 0. 1.]