#load_iris - CNN

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
# from sklearn.datasets import load_iris #load_iris 꽃 확인

#1. 데이터
# Attribute Information (in order):
#     0    - sepal length in cm    꽃받침 길이(Sepal Length)
#     1    - sepal width in cm     꽃받침 폭(Sepal Width)
#     2    - petal length in cm    꽃잎 길이(Petal Length)
#     3    - petal width in cm     꽃잎 폭(Petal Width)
#     4     - class:               setosa, versicolor, virginica의 세가지 붓꽃 종(species)
                # 0 - Iris-Setosa     
                # 1 - Iris-Versicolour
                # 2 - Iris-Virginica

# dataset = load_iris()

# x = dataset.data #(150,4)
# y = dataset.target #(150,)
# #x의 데이터로 세가지 붓꽃 종 중 하나를 찾는 데이터셋이다

# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8) 

x_train = np.load('./data/iris_x_train.npy')
x_test = np.load('./data/iris_x_test.npy')
y_train = np.load('./data/iris_y_train.npy')
y_test = np.load('./data/iris_y_test.npy')


x_pred = x_test[:10]
x_test = x_test[10:]
y_real = y_test[:10]
y_test = y_test[10:]

# print(x_test.shape) #(20, 4)
# print(y_test.shape) #(20,)
# print(x_train.shape) #(120,4)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train_minmax = scaler.transform(x_train) 
x_test_minmax = scaler.transform(x_test)
x_pred_minmax = scaler.transform(x_pred)

x_train_minmax = x_train_minmax.reshape(120,4,1,1)
x_test_minmax = x_test_minmax.reshape(20,4,1,1)
x_pred_minmax = x_pred.reshape(10,4,1,1)
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(32, (2, 2), padding='same', activation='relu', input_shape=(4, 1, 1)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(256, (2, 2), padding='same', activation='relu'))
model.add(Dropout(0.2))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax')) 
model.summary()

model.save('./save/iris_CNN_model.h5')

#3. 컴파일, 훈련
modelpath = './model/iris_CNN-{epoch:02d}-{val_loss:.4f}.hdf5' # hdf5의 파일, {epoch:02d} - epoch의 2자리의 정수, {val_loss:.4f} - val_loss의 소수넷째자리까지가 네이밍됨

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint #ModelCheckpoint 추가
early_stopping = EarlyStopping(monitor='val_loss', patience=100) 
check_point = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto') #val_loss가 가장 좋은 값을 저장할 것이다

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train_minmax, y_train, epochs=1000, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping, check_point])

model.save('./save/iris_CNN_model_fit.h5')
model.save_weights('./save/iris_CNN_model_weight.h5')

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

#4. 평가, 예측
result = model.evaluate(x_test_minmax, y_test, batch_size=32)
print("loss : ", result[0])
print("acc : ", result[1])

y_predict = model.predict(x_pred_minmax)
y_pred = np.argmax(y_predict, axis=1) 

print("y_real : ", y_real)
print("y_pred : ", y_pred)

# 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6)) #단위 무엇인지 찾아보기
plt.subplot(2,1,1) #2장(2행1열) 중 첫번째
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplot(2,1,2) #2장(2행1열) 중 두번째
plt.plot(hist.history['acc'], marker='.', c='red')
plt.plot(hist.history['val_acc'], marker='.', c='blue')
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.show()

# 결과값
# loss :  0.013886451721191406
# acc :  1.0
# y_real :  [0 0 2 1 0 2 0 1 1 2]
# y_pred :  [0 0 2 2 0 2 0 2 2 2]