#ModelCheckPoint - 36_minist_2_CNN.py

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


#1_1. 데이터 전처리 - OneHotEncoding
from tensorflow.keras.utils import to_categorical #keras
#from sklearn.preprocessing import OneHotEncoder #sklearn
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape) #(60000, 10)
print(y_train[0]) #5 - [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.] - 0/1/2/3/4/5/6/7/8/9 - 5의 위치에 1이 표시됨

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255. 
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255.
x_predict = x_predict.reshape(10,28,28,1).astype('float32')/255

#2. 모델구성 
# model = Sequential()
# model.add(Conv2D(10, (2,2), padding='same', input_shape=(28,28,1))) #(28,28,10)
# model.add(Conv2D(20, (2,2), padding='valid')) #(27,27,20)
# model.add(Conv2D(30, (3,3))) #(25,25,30)
# model.add(Conv2D(40, (2,2), strides=2)) #(24,24,40)
# model.add(MaxPooling2D(pool_size=2))
# model.add(Flatten()) 
# model.add(Dense(100, activation='relu')) 
# model.add(Dense(10, activation='softmax')) 
# model.summary()

from tensorflow.keras.models import load_model
model = load_model('./save/model_test01_1.h5') #훈련다시들어감!! 중요
model.summary()

#3. 컴파일, 훈련
modelpath = './model/mnist-{epoch:02d}-{val_loss:.4f}.hdf5' # hdf5의 파일, {epoch:02d} - epoch의 2자리의 정수, {val_loss:.4f} - val_loss의 소수넷째자리까지가 네이밍됨

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint #ModelCheckpoint 추가
early_stopping = EarlyStopping(monitor='val_loss', patience=10) 
check_point = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto') #val_loss가 가장 좋은 값을 저장할 것이다

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=30, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping, check_point])

# loss = hist.history['loss']
# val_loss = hist.history['val_loss']
# acc = hist.history['acc']
# val_acc = hist.history['val_acc']

#4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", result[0])
print("acc : ", result[1])

y_pred = model.predict(x_predict)
y_pred = np.argmax(y_pred, axis=1) 
print("y_col : ", y_col)
print("y_pred : ", y_pred)

# # 시각화
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10,6)) #단위 무엇인지 찾아보기
# plt.subplot(2,1,1) #2장(2행1열) 중 첫번째
# plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
# plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
# plt.grid()
# plt.title('loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc='upper right')

# plt.subplot(2,1,2) #2장(2행1열) 중 두번째
# plt.plot(hist.history['acc'], marker='.', c='red')
# plt.plot(hist.history['val_acc'], marker='.', c='blue')
# plt.grid()
# plt.title('acc')
# plt.ylabel('acc')
# plt.xlabel('epoch')
# plt.legend(['acc', 'val_acc'])

# plt.show()

# 결과값
# loss :  0.07919305562973022
# acc :  0.9842000007629395
# y_col :  [7 2 1 0 4 1 4 9 5 9]
# y_pred :  [7 2 1 0 4 1 4 9 5 9]