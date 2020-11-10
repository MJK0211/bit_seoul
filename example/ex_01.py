import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1,101))
y = np.array(range(1,101))

x_train_org, x_test, y_train_org, y_test = train_test_split(x, y, train_size=0.7)
x_train, x_val, y_train, y_val = train_test_split(x_train_org, y_train_org, train_size=0.7)

#모델
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(3))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val))

#평가, 예측
loss = model.evaluate(x_train, y_train, batch_size=1)
print("loss : ", loss)

y_pred = model.predict(x_test)
print("y_pred : \n", y_pred)

from sklearn.metrics import mean_absolute_error

def RMSE(y_test, y_pred):
    return np.sqrt(mean_absolute_error(y_test, y_pred))

print("RMSE : ", RMSE(y_test, y_pred))

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)

print("R2: ", r2)