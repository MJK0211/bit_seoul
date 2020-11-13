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

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard #TensorBoard 추가
early_stopping = EarlyStopping(monitor='loss', patience=200, mode='min') 
to_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True) 
#graph 폴더에 파일들이 생기는 것을 볼 수 있다
#cmd - d: - cd Study - cd bit_seoul - cd graph (파일 최종 경로) - tensorboard --logdir=. enter
#TensorBoard 2.3.0 at http://localhost:6006/ (Press CTRL+C to quit)
#위에 표시된 url 접속
#log가 겹칠 수 있다. 새로 볼 경우에는 로그를 지우고 새로 하는게 좋을 수 있다

history = model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split= 0.2, verbose=1, callbacks=[early_stopping, to_hist]) #to_hist를 콜백에 list형태로 추가

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("mae : ", mae)

y_predict = model.predict(x_test)
print("y_predict : \n", y_predict)