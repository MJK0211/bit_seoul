#36_mnist2_CNN 버전을 LSTM 코딩하기
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LSTM
from tensorflow.keras.datasets import mnist #dataset인 mnist추가

#1. 데이터
#mnist를 통해 손글씨 1~9까지의 데이터를 사용

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_predict = x_test[:10]
y_col = y_test[:10]
print(x_predict.shape) #(10,28,28)


#1_1. 데이터 전처리 - OneHotEncoding
from tensorflow.keras.utils import to_categorical #keras
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape) #(60000, 10)
print(y_train[0]) #5 - [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.] - 0/1/2/3/4/5/6/7/8/9 - 5의 위치에 1이 표시됨

x_train = x_train.reshape(60000,14,56).astype('float32')/255. 
x_test = x_test.reshape(10000,14,56).astype('float32')/255.
x_predict = x_predict.reshape(10,14,56).astype('float32')/255

#2. 모델구성 
model = Sequential()
model.add(LSTM(5, activation='relu', input_shape=(14,56))) 
model.add(Dense(10, activation='softmax')) 
model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) # mse - 'mean_squared_error' 가능

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='min') 
# to_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True) 
model.fit(x_train, y_train, epochs=30, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping])

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", loss)
print("acc : ", acc)

y_pred = model.predict_classes(x_predict) # np.argmax로 변환하지 않아도 같은 값으로 출력이 가능하다
#y_pred = np.argmax(y_pred, axis=1)
print("y_col : ", y_col)
print("y_pred : ", y_pred)

# 결과값
# loss :  0.263659805059433
# acc :  0.9233999848365784
# y_col :  [7 2 1 0 4 1 4 9 5 9]
# y_pred :  [7 2 1 0 4 1 4 2 5 9]