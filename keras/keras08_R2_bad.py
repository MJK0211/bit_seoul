#R2를 음수가 아닌 0.5 이하로 줄여보자
#레이어는 인풋과 아웃품을 포함 7개 이상으로 설정 (히든이 5개 이상)
#히든레이어 노드는 레이어당 각각 최소 10개 이상
#batch_size = 1
#epochs = 100 이상
#데이터 조작 금지

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
model.add(Dense(100000, input_dim=1)) 
model.add(Dense(1000))
model.add(Dense(10000))
model.add(Dense(100))
model.add(Dense(10000))  
model.add(Dense(1)) 

#일반적으로 사각형, 다이아몬드, 역삼각형의 구조를 가진다. 반대로 낮은 R2값을 얻기 위해서 지그재그 형태의 레이어를 구성해보니 0.5이하의 R2값을 얻을 수 있었다.

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 
model.fit(x_train, y_train, epochs=100, batch_size=1)

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
# loss:  1.273236632347107
# result :
#  [[10.219851]
#  [11.057535]
#  [11.895217]
#  [12.732902]
#  [13.570585]]
# RMSE :  1.1283779230881017
# R2:  0.36338163134369095
