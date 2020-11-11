#실습 train_test_split을 행렬의 슬라이싱으로 바꿀것

#1. 데이터
import numpy as np

x = np.array([range(1,101), range(711,811), range(100)]) #100개의 데이터 3개 - 100행 3열
y = np.array([range(101,201), range(311,411), range(100)])

# print(x)
# print(x.shape) #(3,100)

x = np.transpose(x)
y = np.transpose(y)

#train_test_split 사용하지 않고 slicing 해서 사용하기
x_train = x[:60]
y_train = y[:60]

x_val = x[60:80]
y_val = y[60:80]

x_test = x[80:100]
y_test = y[80:100]

#2. 모델구성
from tensorflow.keras.models import Sequential #순차적모델
from tensorflow.keras.layers import Dense #가장 기본적인 모델인 Dense 사용

model = Sequential()
model.add(Dense(10, input_dim=3)) #3가지 input
model.add(Dense(5))
model.add(Dense(3)) #따라서 3개의 아웃풋이 나와야 한다

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1, validation_data=(x_val, y_val))

#4. 평가, 예측
loss = model.evaluate(x_train, y_train, batch_size=1)
y_pred = model.predict(x_test)

print("y_test : \n", y_test)
print("y_pred : \n", y_pred)

from sklearn.metrics import mean_absolute_error
def RMSE(y_test, y_pred):
    return np.sqrt(mean_absolute_error(y_test, y_pred))
print("RMSE : ", RMSE(y_test, y_pred))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R2 : ", r2)