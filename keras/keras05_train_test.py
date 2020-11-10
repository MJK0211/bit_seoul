from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x_pred = np.array([11,12,13])

#2. 모델구성
model = Sequential() 
model.add(Dense(30, input_dim=1)) 
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(7))
model.add(Dense(1)) 

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y, batch_size=1)
print("loss: ", loss)

y_pred = model.predict(x_pred)
print("result : \n", y_pred)

# 1. 결과값
# loss:  4.959588512445934e-13
# result :
#  [[10.999999]
#  [12.      ]
#  [12.999999]]

# 2. 결과값
# loss:  1.0928147492830775e-12
# result :
#  [[10.999999]
#  [11.999999]
#  [12.999999]]