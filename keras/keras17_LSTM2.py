import numpy as np  

#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])   #(4,3) 
y = np.array([4,5,6,7]) #(4,)

x = x.reshape(x.shape[0], x.shape[1], 1) 
print("x.shape: ", x.shape) #(4,3,1)
print("y.shape: ", y.shape)
print(x)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM 

model = Sequential()
# model.add(LSTM(10, activation='relu', input_shape=(3,1))) 
model.add(LSTM(10, activation='relu', input_length=3, input_dim=1)) # input_shape(3,1)과 같다
model.add(Dense(20))
model.add(Dense(5))
model.add(Dense(1))
model.summary()


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

