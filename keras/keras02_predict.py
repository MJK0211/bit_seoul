from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

#2. 모델구성
model = Sequential() 
model.add(Dense(3, input_dim=1)) 
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1)) 

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측
loss, acc = model.evaluate(x, y, batch_size=1)
print("loss: ", loss)
print("acc: ", acc)

y_pred = model.predict(x)
print("result : ", y_pred)

#accuracy에서 머신이 생각하는 결과값은 1과 0.999는 다르다고 생각한다. 
#따라서 선형회귀에서는 accuracy라는 평가 지표는 사용할 수 없다.
#So 선형회귀에서는 loss값과 결과값을 확인해보면 된다.




