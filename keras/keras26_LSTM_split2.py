#1~100까지의 데이터를 LSTM으로 훈련 및 예측
#train, test 분리
#ealry_stopping 사용

import numpy as np   

#1. 데이터
dataset = np.array(range(1,101))
size = 5

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset]) 
    # print(type(aaa))
    return np.array(aaa)

datasets = split_x(dataset, size)

bbb = []
ccc = []
for i in range(len(datasets)):    
    bbb.append(datasets[i][0:4])
    ccc.append(datasets[i][4:5])
x = np.array(bbb) #(96,4)
y = np.array(ccc) #(96,1)

x = x.reshape(x.shape[0], x.shape[1], 1) #(96,4,1)
x_pred = np.array([97,98,99,100]) #(4,)
x_pred = x_pred.reshape(1,4,1) #(1,4,1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, = train_test_split(x, y, train_size=0.7) 

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM 

model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(4,1)))
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

y_predict = model.predict(x_pred)
print("y_predict : \n", y_predict)

# 결과값
# loss :  0.00010231292253592983
# y_predict :
#  [[100.98764]]
