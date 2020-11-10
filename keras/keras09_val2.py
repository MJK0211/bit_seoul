from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#1. 데이터
x_train = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]) #훈련시킬 데이터
y_train = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]) 

# x_val = np.array([11,12,13,14,15]) #검증할 데이터
# y_val = np.array([11,12,13,14,15])

x_test = np.array([16,17,18,19,20]) #테스트할 데이터
y_test = np.array([16,17,18,19,20])

#2. 모델구성
model = Sequential() 
model.add(Dense(30, input_dim=1)) 
model.add(Dense(500))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1)) 

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 
model.fit(x_train, y_train, epochs=100, validation_split=0.2) 

#validation_split
#검증은 하겠으나 train의 20프로를 잡겠다. ex) 15개이면 12개는 train 3개는 validation

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss: ", loss)

y_pred = model.predict(x_test)
print("result : \n", y_pred)

from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

print("RMSE : ", RMSE(y_test, y_pred))

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)

print("R2: ", r2)