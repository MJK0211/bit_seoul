#Data 전처리 - 매우 중요

from numpy import array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input

#1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], 
           [5,6,7], [6,7,8], [7,8,9], [8,9,10],
           [9,10,11], [10,11,12], 
           [2000,3000,4000], [3000,4000,5000], [4000,5000,6000],
           [100,200,300]]) # (14,3)

y = array([4,5,6,7,8,9,10,11,12,13,5000,6000,7000,400]) #(14,)
#전처리에서는 y값은 target이기 때문에 건드리지 않는다
#x값이 변하도 결과값은 같기 때문
#연산에는 전혀 문제가 없다
x_predict = array([55,65,75]) #(3,)
x_predict = x_predict.reshape(1,3)
# print(x_predict)

x_predict2 = array([6600,6700,6800]) #전처리 범위에서 벗어난 데이터를 쓸 경우?
# x_predict2 = x_predict2.reshape(1,3)

print(x_predict)
from sklearn.preprocessing import MinMaxScaler #데이터 전처리 MinMaxScaler 추가

scaler = MinMaxScaler()
scaler.fit(x)
x_minmax = scaler.transform(x) #자동으로 min, max를 검출하여 데이터를 전처리 해줌
x_pred_minmax = scaler.transform(x_predict)
# x_pred2_minmax = scaler.transform(x_predict2)

# print(x_pred_minmax)
print(x_minmax)
print(x_pred_minmax)

#x_predict : [[55 65 75]]
#x_pred_minmax : [[0.01350338 0.01260504 0.012006  ]]

# x_minmax
# 결과값
# [[0.00000000e+00 0.00000000e+00 0.00000000e+00]  -> 1은 0
#  [2.50062516e-04 2.00080032e-04 1.66750042e-04]  -> 열마다 데이터 최대값이 다름 1번째 열은 4000, 3번째 열은 6000
#  [5.00125031e-04 4.00160064e-04 3.33500083e-04]
#  [7.50187547e-04 6.00240096e-04 5.00250125e-04]
#  [1.00025006e-03 8.00320128e-04 6.67000167e-04]
#  [1.25031258e-03 1.00040016e-03 8.33750208e-04]
#  [1.50037509e-03 1.20048019e-03 1.00050025e-03]
#  [1.75043761e-03 1.40056022e-03 1.16725029e-03]
#  [2.00050013e-03 1.60064026e-03 1.33400033e-03]
#  [2.25056264e-03 1.80072029e-03 1.50075038e-03]
#  [4.99874969e-01 5.99839936e-01 6.66499917e-01]
#  [7.49937484e-01 7.99919968e-01 8.33249958e-01]
#  [1.00000000e+00 1.00000000e+00 1.00000000e+00]  -> 열기준 4000, 5000, 6000짜리가 1 (최대값이기 때문)
#  [2.47561890e-02 3.96158463e-02 4.95247624e-02]] -> 전체 데이터는 0~1사이의 값으로 전처리 된것을 볼 수 있다


# x_minmax = x_minmax.reshape(x_minmax.shape[0], x_minmax.shape[1], 1)
# x_pred_minmax = x_pred_minmax.reshape(1,3) 
# x_pred2_minmax = x_pred2_minmax.reshape(1,3,1) 


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, = train_test_split(x_minmax, y, train_size=0.8) 

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM 

model = Sequential()
model.add(Dense(200, activation='relu', input_shape=(3,)))
model.add(Dense(180, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(110, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping #EarlyStopping 추가 - 조기종료
early_stopping = EarlyStopping(monitor='loss', patience=200, mode='min') 

model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.25, verbose=1, callbacks=[early_stopping])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_pred_minmax)
print("y_predict : \n", y_predict)

# LSTM으로 했을때
# 결과값
# loss :  81165.265625
# y_predict :
#  [[8.222445]]
# 너무 상이한 결과값이나옴.. 이유는 찾지 못함

# Sequencial Dense층으로만 구성, metrics 'mae', validation_split=0.25 추가
# loss :  [14.905375480651855, 3.6674740314483643]
# y_predict :
#  [[81.92779]]