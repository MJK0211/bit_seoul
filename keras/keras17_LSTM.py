import numpy as np  

#1. 데이터

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])   #(4,3) 
y = np.array([4,5,6,7]) #(4,)
#[1,2,3] - 4, [2,3,4] - 5, [3,4,5] - 5, [4,5,6] - 7을 예측하겠다.
# x의 결과
# [[1 2 3]
#  [2 3 4]
#  [3 4 5]
#  [4 5 6]]

x = x.reshape(x.shape[0], x.shape[1], 1) #행, 렬, 몇개씩 자를건지에 대해 reshape해줌
# x = x.reshape(4,3,1) 위와 같음

print("x.shape: ", x.shape) #(4,3,1)
print("y.shape: ", y.shape)
print(x)
# x결과
# [[[1]
#   [2]
#   [3]]
#  [[2]
#   [3]
#   [4]]
#  [[3]
#   [4]
#   [5]]
#  [[4]
#   [5]
#   [6]]]

#2. 모델구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM #LSTM 사용

model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(3,1))) 
#input_shape가 (3,1)인 이유 - LSTM은 3차원에 데이터로 존재해야한다 따라서 (4,3,1) 형태에서 행무시! - (3,1), 1개씩 작업
#행, 렬, 몇개씩자르는지
model.add(Dense(20))
model.add(Dense(5))
model.add(Dense(1))
model.summary()

# 1 - 12, 2 - 32, 3 - 60, 5 - 140, 10 - 480
# 결과
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# lstm (LSTM)                  (None, 10)                480           -> 초기행렬 (4,3) - Data 4개, 1개씩 검사
# _________________________________________________________________    -> 4(1개씩 검사(1) + bias(1) + 노드의 갯수(10)) * 노드의 갯수(10) = 480
# dense (Dense)                (None, 20)                220
# _________________________________________________________________
# dense_1 (Dense)              (None, 10)                210
# _________________________________________________________________
# dense_2 (Dense)              (None, 1)                 11
# =================================================================
# Total params: 921
# Trainable params: 921
# Non-trainable params: 0
# _________________________________________________________________

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1, validation_split=0.25)

x_input = np.array([5,6,7]) # (3,) -> (1,3,1)
x_input = x_input.reshape(1,3,1)

#4. 평가, 예측

loss = model.evaluate(x, y, batch_size=1)
print("loss : ", loss)
y_pred = model.predict(x_input)

print("y_pred : \n", y_pred)

# 결과값
# result 가중치 때문에 돌릴때마다 결과값이 다르게 나온다
# 1. 결과
# loss :  0.017552146688103676
# y_pred :
#  [[8.004369]]
# 2. 결과
# loss :  0.09580722451210022
# y_pred :
#  [[8.517853]]