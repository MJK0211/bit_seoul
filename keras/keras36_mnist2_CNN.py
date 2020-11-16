#OneHotEncodeing

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.datasets import mnist #dataset인 mnist추가

#1. 데이터
#mnist를 통해 손글씨 1~9까지의 데이터를 사용

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, x_test.shape) #(60000,28,28), (10000,28,28)
# print(y_train.shape, y_test.shape) #(60000,), (10000,)

x_predict = x_test[:10]
y_col = y_test[:10]
print(x_predict.shape) #(10,28,28)
# plt.imshow(x_train[0], 'winter_r')
# plt.show()


#1_1. 데이터 전처리 - OneHotEncoding
from tensorflow.keras.utils import to_categorical #keras
#from sklearn.preprocessing import OneHotEncoder #sklearn
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape) #(60000, 10)
print(y_train[0]) #5 - [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.] - 0/1/2/3/4/5/6/7/8/9 - 5의 위치에 1이 표시됨

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255. #CNN은 4차원이기 때문에 4차원으로 변환, astype -0 형변환
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255.
x_predict = x_predict.reshape(10,28,28,1).astype('float32')/255

#2. 모델구성 
model = Sequential()
model.add(Conv2D(10, (2,2), padding='same', input_shape=(28,28,1))) #(28,28,10)
model.add(Conv2D(20, (2,2), padding='valid')) #(27,27,20)
model.add(Conv2D(30, (3,3))) #(25,25,30)
model.add(Conv2D(40, (2,2), strides=2)) #(24,24,40)
model.add(MaxPooling2D(pool_size=2)) #기본 Default는 2이다 - (12,12,40)
model.add(Flatten()) #현재까지 내려왔던 것을 일자로 펴주는 기능 - 이차원으로 변경 (12*12*40 = 1440) = (1440,) 다음 Dense층과 연결시키기 위해 사용
model.add(Dense(100, activation='relu')) # CNN은 activation default = 'relu', LSTM activation default='tanh'
model.add(Dense(10, activation='softmax')) 
model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) # mse - 'mean_squared_error' 가능

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='min') 
to_hist = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True) 
model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping])

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", loss)
print("acc : ", acc)

print(x_predict)
y_pred = model.predict(x_predict)
y_pred = np.argmax(y_pred, axis=1).reshape(-1,1)
print("y_col : ", y_col)
print("y_pred : ", y_pred)

# 결과값
# loss :  0.0910315066576004
# acc :  0.9884999990463257

#실습1. test데이터를 10개 가져와서 predict 만들것    
#실습2. 모델: early_stopping 적용, tensorboard 추가

# y_col :  [7 2 1 0 4 1 4 9 5 9]
# y_pred :  [[7]
#  [2]
#  [1]
#  [0]
#  [4]
#  [1]
#  [4]
#  [9]
#  [5]
#  [9]]