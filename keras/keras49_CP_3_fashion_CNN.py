#Fashion_Mnist CNN

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
# from tensorflow.keras.datasets import fashion_mnist #dataset인 fashion_mnist 추가
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

# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = np.load('./data/fashion_mnist_x_train.npy')
x_test = np.load('./data/fashion_mnist_x_test.npy')
y_train = np.load('./data/fashion_mnist_y_train.npy')
y_test = np.load('./data/fashion_mnist_y_test.npy')


x_predict = x_test[:10]
x_test = x_test[10:]
y_real = y_test[:10]
y_test = y_test[10:]

print(x_train.shape) #(60000, 28, 28)
print(x_test.shape) #(9990, 28, 28)
print(y_test.shape) #(9990, )

#1_1. 데이터 전처리 - OneHotEncoding

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255. 
x_test = x_test.reshape(9990,28,28,1).astype('float32')/255.
x_predict = x_predict.reshape(10,28,28,1).astype('float32')/255

#2. 모델구성 
model = Sequential()
model.add(Conv2D(10, (2,2), padding='same', input_shape=(28,28,1))) #(28,28,10)
model.add(Conv2D(20, (2,2), padding='valid')) #(27,27,20)
model.add(Conv2D(30, (3,3))) #(25,25,30)
model.add(Conv2D(40, (2,2), strides=2)) #(24,24,40)
model.add(MaxPooling2D(pool_size=2)) #(12,12,40)
model.add(Flatten()) 
model.add(Dense(100, activation='relu')) 
model.add(Dense(10, activation='softmax')) 
model.summary()

model.save('./save/fashion_CNN_model.h5')

#3. 컴파일, 훈련
modelpath = './model/fashion_CNN-{epoch:02d}-{val_loss:.4f}.hdf5' # hdf5의 파일, {epoch:02d} - epoch의 2자리의 정수, {val_loss:.4f} - val_loss의 소수넷째자리까지가 네이밍됨
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint #ModelCheckpoint 추가
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min') 
check_point = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='min') #val_loss가 가장 좋은 값을 저장할 것이다

hist = model.fit(x_train, y_train, epochs=30, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_stopping, check_point])

model.save('./save/fashion_CNN_model_fit.h5')
model.save_weights('./save/fashion_CNN_model_weight.h5')

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
# loss :  0.3772709369659424
# acc :  0.8958958983421326
# y_real :  [9 2 1 1 6 1 4 6 5 7]
# y_pred :  [9 2 1 1 6 1 4 6 5 7]