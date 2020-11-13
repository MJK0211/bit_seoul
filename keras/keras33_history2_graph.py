import numpy as np   

#1. 데이터
dataset = np.array(range(1,11))
size = 5

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset]) 
    # print(type(aaa))
    return np.array(aaa)

datasets = split_x(dataset, size)

x = datasets[:, 0:4] #(96,4)
y = datasets[:, 4] #(96,)
x = np.reshape(x, (x.shape[0], x.shape[1],1))  #(96,4,1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, = train_test_split(x, y, train_size=0.9, shuffle=False) 

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
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping #EarlyStopping 추가 - 조기종료
early_stopping = EarlyStopping(monitor='loss', patience=200, mode='min') 

history = model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split= 0.2, verbose=1, callbacks=[early_stopping]) #history 설정

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("mae : ", mae)

y_predict = model.predict(x_test)
print("y_predict : \n", y_predict)

#그래프 - matplotlib
import matplotlib.pyplot as plt 

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('loss & mae')
plt.ylabel('loss & mae')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss', 'train mae', 'val mae']) #그래프에 대한 명시
plt.show() #plt완성 및 보여주기

#history를 가지고 early_stopping에 대해서 세부적으로 확인하고 변경할 수 있다
#history는 데이터의 시각화이다 

