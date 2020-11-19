#load_breast_cancer - DNN - 이진분류(유방암) 걸렸으면1, 아니면2

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
# from sklearn.datasets import load_breast_cancer #load_breast_cancer 이진분류(유방암) 걸렸으면1, 아니면2

# dataset = load_breast_cancer()

# x = dataset.data #(569, 30)
# y = dataset.target #(569,)

# print(x.shape) #(559, 30)

# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test, = train_test_split(x, y, train_size=0.8) 

x_train = np.load('./data/cancer_x_train.npy')
x_test = np.load('./data/cancer_x_test.npy')
y_train = np.load('./data/cancer_y_train.npy')
y_test = np.load('./data/cancer_y_test.npy')

x_predict = x_test[:10]
x_test = x_test[10:]
y_real = y_test[:10]
y_test = y_test[10:]

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train_minmax = scaler.transform(x_train) 
x_test_minmax = scaler.transform(x_test)
x_pred_minmax = scaler.transform(x_predict)


#2. 모델 구성
model = Sequential()
model.add(Dense(32, input_shape=(30,))) 
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu')) 
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu')) 
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid')) 
model.summary()

model.save('./save/cancer_DNN_model.h5')

#3. 컴파일, 훈련
modelpath = './model/cancer_DNN-{epoch:02d}-{val_loss:.4f}.hdf5' # hdf5의 파일, {epoch:02d} - epoch의 2자리의 정수, {val_loss:.4f} - val_loss의 소수넷째자리까지가 네이밍됨
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint #ModelCheckpoint 추가
early_stopping = EarlyStopping(monitor='val_loss', patience=100, mode='auto') 
check_point = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto') #val_loss가 가장 좋은 값을 저장할 것이다

hist = model.fit(x_train_minmax, y_train, epochs=1000, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping, check_point])

model.save('./save/cancer_DNN_model_fit.h5')
model.save_weights('./save/cancer_DNN_model_weight.h5')

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

#4. 평가, 예측
result = model.evaluate(x_test_minmax, y_test, batch_size=32)
print("loss : ", result[0])
print("acc : ", result[1])

y_predict = model.predict(x_pred_minmax)
y_predict = np.round(y_predict.reshape(10,),0)
print("y_real : ", y_real)
print("y_pred : ", y_predict)

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
# loss :  0.018005581572651863
# acc :  0.9903846383094788
# y_real :  [1 0 1 1 0 0 0 1 1 1]
# y_pred :  [1. 1. 1. 1. 0. 0. 1. 1. 1. 1.]