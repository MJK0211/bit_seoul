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
model.compile(loss='mse', optimizer='adam', metrics=['acc', 'mse', 'mae']) #metrics 'acc', 'mse', 'mae 사용, 최종 evaluate의 반환값에 추가
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y, batch_size=1) #loss와 metrics를 리스트 형태로 loss에 반환
#loss:  [1.0022915657159626e-11, 0.10000000149011612, 1.0022915657159626e-11, 2.5391577764821704e-06]
             
print("loss: ", loss)
#print("acc: ", acc)







