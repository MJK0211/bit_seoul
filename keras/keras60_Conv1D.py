import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from tensorflow.keras.layers import MaxPooling1D

#1. 데이터
a = np.array(range(1,100))
size = 5

#splix_x 멋진 함수를 데려오고
def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset]) 
    # print(type(aaa))
    return np.array(aaa)

datasets = split_x(a, size)

x = datasets[0:,:4] #(95, 4)
y = datasets[0:,4:] #(95, 1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=False)

print(x_train.shape) #(66, 4) 
print(x_test.shape) #(29, 4)

x_predict = x_test[:10]
x_test = x_test[10:]
y_real = y_test[:10]
y_test = y_test[10:]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train_minmax = scaler.transform(x_train)
x_test_minmax = scaler.transform(x_test)
x_predict_minmax = scaler.transform(x_predict)

x_train_minmax = x_train_minmax.reshape(66,2,2)
x_test_minmax = x_test_minmax.reshape(19,2,2)
x_pred_minmax = x_predict_minmax.reshape(10,2,2)

#2. 모델 구성
model = Sequential()
model.add(Conv1D(32, (2), padding='same', input_shape=(2,2))) #(28,28,10)
model.add(Conv1D(64, (2), padding='same'))
model.add(MaxPooling1D()) #(12,12,40)
model.add(Flatten()) 
model.add(Dense(100, activation='relu')) 
model.add(Dense(1))
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='min') 
model.fit(x_train_minmax, y_train, epochs=100, batch_size=4, verbose=1, validation_split=0.2, callbacks=[early_stopping])

#4. 평가, 예측
loss = model.evaluate(x_test_minmax, y_test, batch_size=4)
print("loss : ", loss)

y_predict = model.predict(x_pred_minmax)
print("y_real : ", y_real.reshape(10,))
print("y_pred : ", y_predict.reshape(10,))

# 결과값

# loss :  0.002308957278728485
# y_real :  [71 72 73 74 75 76 77 78 79 80]
# y_pred :  [70.978424 71.97704  72.975685 73.97435  74.97302  75.97169  76.97037  77.96904  78.967705 79.96638 ]