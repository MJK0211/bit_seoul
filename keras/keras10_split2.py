from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#1. 데이터
x = np.array(range(1,101))
y = np.array(range(101,201))

x_train = x[:60] #60개 
y_train = y[:60]

x_val = x[60:80] #20개
y_val = y[60:80] 

x_test = x[80:] #20개
y_test = y[80:]

#2. 모델구성
model = Sequential() 
model.add(Dense(150, input_dim=1)) 
model.add(Dense(300))
model.add(Dense(500))
model.add(Dense(150))
model.add(Dense(1)) 

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 
model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val)) 

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

# 결과값
# RMSE :  0.2988924539270431
# R2:  0.9973131819845255