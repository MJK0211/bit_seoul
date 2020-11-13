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
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
from tensorflow.keras.callbacks import EarlyStopping #EarlyStopping 추가 - 조기종료
early_stopping = EarlyStopping(monitor='loss', patience=200, mode='min') 

history = model.fit(x_train, y_train, epochs=7, batch_size=1, validation_split= 0.2, verbose=1, callbacks=[early_stopping]) #history 설정

print(history) #<tensorflow.python.keras.callbacks.History object at 0x0000027FCE91A9A0> - history안에있는 ~자료형이다
print("======================================")
print(history.history.keys()) #dict_keys(['loss', 'mse', 'val_loss', 'val_mse']) dict_keys = dictionary 형태 - key/value로 이루어져 있다.
print("======================================")
print(history.history['loss']) #[39.742637634277344, 28.41402816772461, 11.229092597961426, 4.145739555358887, 7.967973232269287, 1.9690494537353516, 4.069993019104004]
                               #epoch 하나당 값을 가지기 때문에 리스트 형태로 출력
print("======================================")
print(history.history['val_loss']) #[59.83634948730469, 26.620304107666016, 3.1341845989227295, 52.036136627197266, 0.837214469909668, 2.4478626251220703, 2.6842923164367676]

'''
#4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("mse : ", mse)

y_predict = model.predict(x_test)
print("y_predict : \n", y_predict)

'''