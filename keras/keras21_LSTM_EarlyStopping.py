#LSTM완성
#예측값 80 만들기

import numpy as np  

#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], 
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12], 
              [20,30,40], [30,40,50], [40,50,60]]) # (13,3)

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70]) #(13,)

x = x.reshape(x.shape[0], x.shape[1], 1) # (13,3,1)

x_input = np.array([50,60,70])
x_input = x_input.reshape(1,3,1)

print(x.shape)
print(y.shape)

#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input

# model = Sequential()
# model.add(LSTM(60, activation='relu', input_length=3, input_dim=1)) # input_shape(3,1)과 같다
# model.add(Dense(1250))
# model.add(Dense(700))
# model.add(Dense(150))
# model.add(Dense(1))
# model.summary()

input1 = Input(shape=(3,1))
lstm_layer = LSTM(200, activation='relu', name='lstm_layer')(input1)
dense1 = Dense(180, activation='relu', name='dense1')(lstm_layer)
dense2 = Dense(150, activation='relu', name='dense2')(dense1)
dense3 = Dense(110, activation='relu', name='dense3')(dense2)
dense4 = Dense(60, activation='relu', name='dense4')(dense3)
dense5 = Dense(10, activation='relu', name='dense5')(dense4)
output1 = Dense(1, name='output1')(dense5) 
model = Model(inputs = input1, outputs = output1) 

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping #EarlyStopping 추가 - 조기종료
early_stopping = EarlyStopping(monitor='loss', patience=200, mode='min') 
#patience 몇번까지 봐줄거냐? 바로 끝내기보다는 조금 더 지켜보고 최소값을 정하겠다. mode는 최소값
#최소값보다 내려가면 계속진행, 올라간다면 멈춤
model.fit(x, y, epochs=10000, batch_size=1, verbose=1, callbacks=[early_stopping])

#4. 평가, 예측
loss = model.evaluate(x, y)
y_pred = model.predict(x_input)

print("loss : ", loss)
print("x_input : \n", x_input)
print("y_pred : \n", y_pred)

# 결과값
# loss :  0.0060346596874296665
# x_input :
#  [[[50]
#   [60]
#   [70]]]
# y_pred :
#  [[80.84698]]