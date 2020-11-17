#Fashion_Mnist LSTM

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LSTM
from tensorflow.keras.datasets import fashion_mnist #dataset인 fashion_mnist 추가
from tensorflow.keras.utils import to_categorical 

#1. 데이터
#fasion_ mnist를 통해 다음과 같은 데이터 셋 사용
# 0 티셔츠/탑
# 1 바지
# 2 풀오버(스웨터의 일종)
# 3 드레스
# 4 코트
# 5 샌들
# 6 셔츠
# 7 스니커즈
# 8 가방
# 9 앵클 부츠

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_predict = x_test[:10]
y_col = y_test[:10]

#1_1. 데이터 전처리 - OneHotEncoding

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,14,56).astype('float32')/255. 
x_test = x_test.reshape(10000,14,56).astype('float32')/255.
x_predict = x_predict.reshape(10,14,56).astype('float32')/255

#2. 모델구성 
model = Sequential()
model.add(LSTM(5, activation='relu', input_shape=(14,56))) 
model.add(Dense(10, activation='softmax')) 
model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='min') 
#to_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True) 
model.fit(x_train, y_train, epochs=30, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping])

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", loss)
print("acc : ", acc)

y_pred = model.predict(x_predict)
y_pred = np.argmax(y_pred, axis=1) 
print("y_col : ", y_col)
print("y_pred : ", y_pred)

# 결과값
# loss :  0.5031107664108276
# acc :  0.8263000249862671
# y_col :  [9 2 1 1 6 1 4 6 5 7]
# y_pred :  [9 2 1 1 6 1 4 6 5 7]