from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#1. 데이터
x_train = np.array([1,2,3,4,5,6,7,8,9,10]) #훈련시킬 데이터
y_train = np.array([1,2,3,4,5,6,7,8,9,10]) 

x_val = np.array([11,12,13,14,15]) #검증할 데이터
y_val = np.array([11,12,13,14,15])

x_test = np.array([16,17,18,19,20]) #테스트할 데이터
y_test = np.array([16,17,18,19,20])

#2. 모델구성
model = Sequential() 
model.add(Dense(30, input_dim=1)) 
model.add(Dense(500))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1)) 

#일반적으로 사각형, 다이아몬드, 역삼각형의 구조를 가진다. 반대로 낮은 R2값을 얻기 위해서 지그재그 형태의 레이어를 구성해보니 0.5이하의 R2값을 얻을 수 있었다.

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 
model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val)) 

#validation_data
#검증 데이터를 훈련할때 추가해준다. 
#검증 데이터를 추가해 줌으로써 test data를 예측하여 새로운 데이터에 대해 얼마나 잘 동작하는지 예측할 수 있을 것이다.

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
# result :
#  [[16.005207]
#  [17.004833]
#  [18.004463]
#  [19.00409 ]
#  [20.003714]]
# RMSE :  0.004492379089214156
# R2:  0.9999899092650594