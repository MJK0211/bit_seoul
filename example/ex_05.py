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
x_predict = x_predict.reshape(1,3,1)

x = x.reshape(x.shape[0], x.shape[1], 1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, = train_test_split(x, y, train_size=0.9, shuffle=False) 

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM 

model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(3,1)))
model.add(Dense(180, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(110, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping #EarlyStopping 추가 - 조기종료
early_stopping = EarlyStopping(monitor='loss', patience=150, mode='min') 

model.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=1, callbacks=[early_stopping])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_predict)
print("y_predict : \n", y_predict)
