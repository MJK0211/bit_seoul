#Cifar100 CNN

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.datasets import cifar100 #dataset인 cifar100추가
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_predict = x_test[:10]
x_test = x_test[10:]
y_real = y_test[:10]
y_test = y_test[10:]

print(x_train.shape) #(50000, 32, 32, 3)
print(x_test.shape) #(9990, 32, 32, 3)
print(y_test.shape) #(9990, 100)
print(y_real.shape) #(10,1)

#1_1. 데이터 전처리 - OneHotEncoding

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.astype('float32')/255. 
x_test = x_test.astype('float32')/255.
x_predict = x_predict.astype('float32')/255

#2. 모델구성 
model = Sequential()
model.add(Conv2D(20, (2,2), padding='same', input_shape=(32,32,3))) #(32,32,20)
model.add(Dropout(0.2))
model.add(Conv2D(40, (2,2), padding='same')) #(32,32,40)
model.add(Dropout(0.2))
model.add(Conv2D(60, (3,3), padding='same')) #(32,32,60)
model.add(Dropout(0.2))
model.add(Conv2D(80, (2,2), strides=2)) #(16,16,80)
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=2)) #(8,8,80)
model.add(Dropout(0.2))
model.add(Flatten()) 
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='softmax')) 
model.summary()

model.save('./save/cifar100_CNN_model.h5')

#3. 컴파일, 훈련
modelpath = './model/cifar100_CNN-{epoch:02d}-{val_loss:.4f}.hdf5' # hdf5의 파일, {epoch:02d} - epoch의 2자리의 정수, {val_loss:.4f} - val_loss의 소수넷째자리까지가 네이밍됨
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint #ModelCheckpoint 추가
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min') 
check_point = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='min') #val_loss가 가장 좋은 값을 저장할 것이다

hist = model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping, check_point])

model.save('./save/cifar100_CNN_model_fit.h5')
model.save_weights('./save/cifar100_CNN_model_weight.h5')

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

#4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", result[0])
print("acc : ", result[1])

y_predict = model.predict(x_predict)
y_predict = np.argmax(y_predict, axis=1)
print("y_real : ", y_real.reshape(10,))
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
# loss :  4.3634233474731445
# acc :  0.3375375270843506
# y_real :  [49 33 72 51 71 92 15 14 23  0]
# y_pred :  [49 65 72 11 71 79 65 74 71  0]