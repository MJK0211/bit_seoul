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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM 

model = Sequential()
# model.add(LSTM(200, activation='relu', input_length=3, input_dim=1))
# model.add(LSTM(200))

model.add(LSTM(200, activation='relu', input_length=3, input_dim=1, return_sequences=True)) #LSTM을 2개이상 엮기 위해서는 return_sequences=True를 설정한다
model.add(LSTM(180, activation='relu', return_sequences=False))
model.add(Dense(180, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(110, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.summary()

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# lstm (LSTM)                  (None, 3, 200)            161600
# _________________________________________________________________
# dense (Dense)                (None, 3, 180)            36180         -> 차원이 그대로 넘어간다
# _________________________________________________________________
# dense_1 (Dense)              (None, 3, 150)            27150
# _________________________________________________________________
# dense_2 (Dense)              (None, 3, 110)            16610
# _________________________________________________________________
# dense_3 (Dense)              (None, 3, 60)             6660
# _________________________________________________________________
# dense_4 (Dense)              (None, 3, 10)             610
# _________________________________________________________________
# dense_5 (Dense)              (None, 3, 1)              11
# =================================================================
# Total params: 248,821
# Trainable params: 248,821
# Non-trainable params: 0
# _________________________________________________________________


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping #EarlyStopping 추가 - 조기종료
early_stopping = EarlyStopping(monitor='loss', patience=200, mode='min') 

model.fit(x, y, epochs=10000, batch_size=1, verbose=1, callbacks=[early_stopping])
#4. 평가, 예측
loss = model.evaluate(x, y)
y_pred = model.predict(x_input)

print("loss : ", loss)
print("x_input : \n", x_input)
print("y_pred : \n", y_pred)

# 결과값
# loss :  0.27401411533355713
# x_input :
#  [[[50]
#   [60]
#   [70]]]
# y_pred :
#  [[79.99343]]

# Value Error
# Sequencail 모델구조에서 LSTM을 연결했을 때 나오는 에러이다
# model.add(LSTM(200, activation='relu', input_length=3, input_dim=1))
# model.add(LSTM(200, activation='relu')) 
# Traceback (most recent call last):
#   File "d:\Study\bit_seoul\keras\keras22_Return_sequences.py", line 28, in <module>
#     model.add(LSTM(200, activation='relu'))
# ValueError('Input ' + str(input_index) + ' of layer ' +
# ValueError: Input 0 of layer lstm_1 is incompatible with the layer: expected ndim=3, found ndim=2. Full shape received: [None, 200]
# lstm_1은 호환되지 않는다. ndim=3 3차원을 기대했으나, ndim2 2차원을 찾았다. 전체 모양은 [None, 200]이다