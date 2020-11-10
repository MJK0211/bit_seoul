from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#1. 데이터
x_train = np.array([1,2,3,4,5,6,7,8,9,10]) #훈련시킬 데이터
y_train = np.array([1,2,3,4,5,6,7,8,9,10]) 

x_test = np.array([11,12,13,14,15]) #평가할 데이터
y_test = np.array([11,12,13,14,15]) 

x_pred = np.array([16,17,18]) #예측할 데이터

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

y_pred = model.predict(x_pred)
print("result : \n", y_pred)

# # 1. 결과값
# loss:  8.684025669936091e-05
# result :
#  [[16.002497]
#  [17.000385]
#  [17.998274]]

# 2. 결과값
# loss:  0.0002704643411561847
# result :
#  [[15.971672]
#  [16.967312]
#  [17.96295 ]]

#같은 코드로 다른 결과를 얻는 것은 훈련에 따라 다르기 때문에 다른 결과값을 얻을 수 있다.