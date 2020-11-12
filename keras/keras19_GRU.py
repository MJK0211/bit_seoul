import numpy as np  

#1. 데이터

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])   #(4,3) 
y = np.array([4,5,6,7]) #(4,)

x = x.reshape(x.shape[0], x.shape[1], 1) 

print("x.shape: ", x.shape) #(4,3,1)
print("y.shape: ", y.shape)
print(x)

#2. 모델구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU #GRU 추가

model = Sequential()
model.add(GRU(10, activation='relu', input_shape=(3,1))) 
model.add(Dense(20))
model.add(Dense(5))
model.add(Dense(1))
model.summary()

# 1 - 12, 2 - 30, 3 - 54, 5 - 120, 10 - 390
# 결과값
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# gru (GRU)                    (None, 10)                390
# _________________________________________________________________
# dense (Dense)                (None, 20)                220       
# _________________________________________________________________
# dense_1 (Dense)              (None, 5)                 105
# _________________________________________________________________
# dense_2 (Dense)              (None, 1)                 6
# =================================================================

'''
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
'''
