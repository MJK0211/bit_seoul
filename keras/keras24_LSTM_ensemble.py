import numpy as np 

#1. 데이터
x1 = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
               [5,6,7], [6,7,8], [7,8,9], [8,9,10],
               [9,10,11], [10,11,12],
               [20,30,40], [30,40,50], [40,50,60]]) #(13,3)

x2 = np.array([[10,20,30], [20,30,40], [30,40,50], [40,50,60,],
               [50,60,70], [60,70,80], [70,80,90], [80,90,100],
               [90,100,110], [100,110,120],
               [2,3,4], [3,4,5], [4,5,6]]) #(13,3)

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70]) #(13,)

x1_predict = np.array([55,65,75]) #(3,)
x2_predict = np.array([65,75,85]) #(3,)

x1 = x1.reshape(13, 3, 1)
x2 = x2.reshape(13, 3, 1)

x1_predict = x1_predict.reshape(1, 3, 1)
x2_predict = x2_predict.reshape(1, 3, 1)


#앙상블모델 완성하기
#결과값 85

#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate, LSTM #concatenate 추가

#2_1 모델1
input1 = Input(shape=(3,1)) #input1 layer 구성
dense1_1 = LSTM(100, activation='relu')(input1)
dense1_2 = Dense(30, activation='relu')(dense1_1) 
dense1_3 = Dense(7, activation='relu')(dense1_2)
output1 = Dense(1)(dense1_3)

#2_2 모델2
input2 = Input(shape=(3,1)) #input2 layer 구성
dense2_1 = LSTM(100, activation='relu')(input2) 
dense2_2= Dense(30, activation='relu')(dense2_1) 
dense2_3= Dense(7, activation='relu')(dense2_2)
output2 = Dense(1)(dense2_3)

#모델 병합
merge1 = Concatenate()([output1, output2]) 

middle1 = Dense(30)(merge1)
middle2 = Dense(7)(middle1)
middle3 = Dense(11)(middle2)

#output 모델 구성 (분기-나눔)
output1_1 = Dense(30)(middle3)
output1_2 = Dense(13)(output1_1)
output1_3 = Dense(7)(output1_2)
output1_4 = Dense(1)(output1_3)

#모델 정의
model = Model(inputs=[input1, input2], outputs=output1_4)
model.summary()


#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping #EarlyStopping 추가 - 조기종료
early_stopping = EarlyStopping(monitor='loss', patience=200, mode='min') 

model.fit([x1, x2], y, epochs=1000, batch_size=1, verbose=1, callbacks=[early_stopping])

#4. 평가, 예측

loss = model.evaluate([x1, x2], y)

y_pred = model.predict([x1_predict, x2_predict])

print("loss : ", loss)
print("x1_predict : \n", x1_predict)
print("x2_predict : \n", x2_predict)
print("y_pred : \n", y_pred)

# 결과값
# loss :  0.38675248622894287
# x1_predict :
#  [[[55]
#   [65]
#   [75]]]
# x2_predict :
#  [[[65]
#   [75]
#   [85]]]
# y_pred :
#  [[85.12566]]