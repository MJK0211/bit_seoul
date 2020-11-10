from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

#2. 모델구성
model = Sequential() 
model.add(Dense(30, input_dim=1)) 
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(7))
model.add(Dense(1)) 

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')#, metrics=['accuracy']) 기본 default 평가지표는 loss를 반환하지만 metrics를 추가하게 되면 evaluate에서는 추가적인 데이터를 리스트형태로 반환
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y, batch_size=1)
print("loss: ", loss)
#print("acc: ", acc)







