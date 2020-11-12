#예측값 80 만들기

import numpy as np  

#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], 
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12], 
              [20,30,40], [30,40,50], [40,50,60]]) # (13,3)

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70]) #(13,)

x_input = np.array([50,60,70])

x = x.reshape(13,3,1)
x_input = x_input.reshape(1,3,1)


print(x.shape) #(13,3)
print(y.shape) #(13,)
print(x_input.shape)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM 

model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(3,1)))
model.add(Dense(180, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(110, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping #EarlyStopping 추가 - 조기종료
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='min') 

model.fit(x, y, epochs=1000, batch_size=1, verbose=1, callbacks=[early_stopping])
#4. 평가, 예측
loss = model.evaluate(x, y)
y_pred = model.predict(x_input)

print("loss : ", loss)
print("x_input : \n", x_input)
print("y_pred : \n", y_pred)

# 결과값 
# loss :  7.135950565338135
# x_input :
#  [[[50]
#   [60]
#   [70]]]
# y_pred :
#  [[75.05443]]