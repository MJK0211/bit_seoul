from tensorflow.keras.datasets import reuters #신문기사 맞추기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=10000, test_split=0.2
    #단어 사전의 갯수 : 10000개이상 써야된다
)

print(x_train.shape, x_test.shape) #(8982,) (2246,) - 8982개의 어절이 존재한다. 조심해야 할 것은 train의 0번째와, 1번째가 길이가 다르다
print(y_train.shape, y_test.shape) #(8982,) (2246,)

print(len(x_train[0])) # 87
print(len(x_train[1])) # 56

# token  = Tokenizer()
# token.fit_on_texts(x_train)
# token.fit_on_texts(x_test)

# print(x_train[0])
# word_size = len(x_train)

# y의 카테고리 값을 출력
category = np.max(y_train) + 1 # 0부터이기 때문에 1을 더해줌
print("카테고리 종류 : ", category) # 카테고리 종류 :  46

# y의 유니크한 값을 출력
y_bumpo = np.unique(y_train)
# y_test = np.unique(y_test)
# print(y_bumpo) 
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]
#(46,)

print(x_train[0])
from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train, padding='pre') #앞에 채운다 : pre, 뒤로 채운다 : post
x_test = pad_sequences(x_test, padding='pre') #앞에 채운다 : pre, 뒤로 채운다 : post

# x_test = pad_sequences(x_test, maxlen=100, padding='pre')


# print(x_train.shape) #(8982, 2376)
# print(x_train[0].shape) #(2376,)
# print(x_train[0]) #(2376,)


# 실습 - 모델구성(Embedding으로)

#2. 모델
model=Sequential()
model.add(Embedding(10000, 100, input_length=1000))
model.add(LSTM(32))
model.add(Dense(100))
model.add(Dense(46, activation='softmax'))
model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es=EarlyStopping(monitor='val_loss', patience=10, mode='auto')
model.fit(x_train, y_train, epochs=100, validation_split=0.2, callbacks=[es])

loss, acc = model.evaluate(x_test, y_test)

print('loss :', loss)
print('acc : ', acc)

# loss : 2.8254828453063965
# acc :  0.6687444448471069
