from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

#2. 모델구성
model = Sequential() 
model.add(Dense(30000, input_dim=1)) 
model.add(Dense(50000))
model.add(Dense(30000))
model.add(Dense(10000)) 
#메모리 필요 크기가 커지기 때문에 실행안됨

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=10000, batch_size=1)

#4. 평가, 예측
loss, acc = model.evaluate(x, y, batch_size=1)
print("loss: ", loss)
print("acc: ", acc)

y_pred = model.predict(x)
print("result : ", y_pred)





