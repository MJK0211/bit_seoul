from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#1. 데이터
x = np.array(range(1,101))
y = np.array(range(101,201))

#train_test_split
from sklearn.model_selection import train_test_split

#train_size를 70 test는 30
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7) #test_size도 있다

# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7 , shuffle=false)
# shuffle(섞다) - default는 true, 섞고 싶지 않다면 false입력, 안쓰면 true

print(x_test)
# x_test 결과값
# [ 89  97  29  85  13  61  80  22  98  39   1  82  35  10   9  67  91  41
#   24  45  72  86  60  49  15 100  93  16  75  46]
# slicing과 다르게 순차적인 데이터값이 아닌 고른 데이터값이 출력된다.
# 순차적인 데이터로 훈련을 한다면 weight값이 제한될 수 있다.
# 따라서 범위를 크게하여 골고루 데이터를 훈련을 시켜 범위를 일정하게 한 후, 범위 안에 나머지 30프로에 대해서 평가를 한다

#2. 모델구성
model = Sequential() 
# model.add(Dense(100, input_dim=1)) 
model.add(Dense(100, input_shape=(1,))) 
model.add(Dense(300))
model.add(Dense(500))
model.add(Dense(150))
model.add(Dense(1)) 

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 
model.fit(x_train, y_train, epochs=100, validation_split=0.2) 

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