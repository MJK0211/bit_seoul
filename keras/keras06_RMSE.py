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

#RMSE(Root Mean Square Error)
#RMSE란 말그대로 실험이나 관측에서 나타나는 오차(Error)를 제곱(Square)해서 평균(Mean)한 값의 제곱근(Root)을 뜻합니다.
#RMSE = root{(e1^2 + e2^2 + … + en^2) / n}
#여기서 e1, e2는 참값과 관측값과의 차 입니다.

#사이킷런이라는 것을 가져오겠다

from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_pred): #x_test를 통해 predict에서 나온 y_pred와 y_test를 비교할 함수 
    return np.sqrt(mean_squared_error(y_test, y_pred))

print("RMSE : ", RMSE(y_test, y_pred))

# 결과값
# loss:  0.0003821266000159085
# result :
#  [[11.022743]
#  [12.021074]
#  [13.019405]
#  [14.017736]
#  [15.016066]]
# RMSE :  0.01954805841117681
