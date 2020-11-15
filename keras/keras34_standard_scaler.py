#Data 전처리 - 매우 중요
#MaxAbs_scaler

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

x_predict = array([55,65,75]) #(3,)
x_predict = x_predict.reshape(1,3)

x_predict2 = array([6600,6700,6800]) #전처리 범위에서 벗어난 데이터를 쓸 경우?

print(x_predict)
from sklearn.preprocessing import MinMaxScaler, StandardScaler #데이터 전처리 StandardScaler 추가

scaler = StandardScaler()
scaler.fit(x)
x_minmax = scaler.transform(x) 
x_pred_minmax = scaler.transform(x_predict)

print(x_minmax)
print(x_pred_minmax)
'''
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
'''