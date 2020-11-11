from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 

import numpy as np 

#1. 데이터

x = np.array(range(1,21))
y = np.array(range(1,21))

from sklearn.model_selection import train_test_split

x_org, x_test, y_org, y_test = train_test_split(x, y, train_size=0.7)
x_train, x_val, y_train, y_val = train_test_split(x_org, y_org, train_size=0.6)

#모델구성
model = Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(300))
model.add(Dense(500))
model.add(Dense(300))
model.add(Dense(1))

#컴파일,훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val), batch_size=1)

#평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=1)
print("Loss : ", loss)

y_pred = model.predict(x_test)
print(x_test)
print("Pred : ", y_pred)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R2 : ", r2)

#11_11 오전 손코딩 완료
