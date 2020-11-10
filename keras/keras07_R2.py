from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#1. 데이터
x_train = np.array([1,2,3,4,5,6,7,8,9,10]) #훈련시킬 데이터
y_train = np.array([1,2,3,4,5,6,7,8,9,10]) 

x_test = np.array([11,12,13,14,15]) #평가할 데이터
y_test = np.array([11,12,13,14,15]) 

#2. 모델구성
model = Sequential() 
model.add(Dense(30, input_dim=1)) 
model.add(Dense(500))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1)) 

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 
model.fit(x_train, y_train, epochs=100)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss: ", loss)

y_pred = model.predict(x_test)
print("result : \n", y_pred)

from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

print("RMSE : ", RMSE(y_test, y_pred))

#R2 결정계수, 회귀 지표이다
#R2값은 회귀 모델에서 예측의 적합도를 0과 1 사이의 값으로 계산한 것이다
#1은 예측이 완벽한 경우고, 0은 훈련 세트의 출력값인 y_train의 평균으로만 예측하는 모델의 경우이다

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)

print("R2: ", r2)

# 결과값
# loss:  0.0020386173855513334
# result :
#  [[10.967549]
#  [11.961595]
#  [12.955639]
#  [13.949689]
#  [14.943733]]
# RMSE :  0.04515105248327365
# R2:  0.9989806912298264 - 1에 근접하기 때문에 잘 나왔다.
# RMSE와 같이 사용한다 - 개인적인 생각으로 가장 근사한 결과값과 예측값의 차이라고 생각



