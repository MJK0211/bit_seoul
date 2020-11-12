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
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU

model = Sequential()
model.add(GRU(60, activation='relu', input_length=3, input_dim=1)) # input_shape(3,1)과 같다
model.add(Dense(1250))
model.add(Dense(700))
model.add(Dense(150))
model.add(Dense(1))
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1, validation_split=0.25)

#4. 평가, 예측
loss = model.evaluate(x, y, batch_size=1)
y_pred = model.predict(x_input)

print("loss : ", loss)
print("x_input : \n", x_input)
print("y_pred : \n", y_pred)

# 결과값
# x_input :
#  [[[50]
#   [60]
#   [70]]]
# y_pred :
#  [[67.352585]]

#         ㅣ         LSTM         ㅣ      SimpleRNN      ㅣ          GRU
# ------------------------------------------------------------------------------
#  80     ㅣ         78.5         ㅣ        79.01        ㅣ         67.35
#  loss   ㅣ        0.0012        ㅣ       0.8945        ㅣ        0.0210
#  params ㅣ   14880 / 1,072,131  ㅣ   3720 / 1,060,971  ㅣ    11340 / 1,068,591