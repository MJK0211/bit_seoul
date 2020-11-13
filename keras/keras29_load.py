#keras26_LSTM_split1.py 에서 구성한 모델을 keras28_save에서 파일로 저장한다.
#keras29_load.py에서 저장한 파일을 불러오고 모델구성을 마무리 한 후 훈련 및 예측하기!

import numpy as np  
from tensorflow.keras.models import load_model #load_model 추가!
from tensorflow.keras.layers import Dense, LSTM

#1. 데이터
dataset = np.array(range(1,101))
size = 5 

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset]) 
    # print(type(aaa))
    return np.array(aaa)

datasets = split_x(dataset, size)

x = datasets[:, 0:4] #for문으로 만들지 않아도 간단히 split이 가능하다 (96,4)
y = datasets[:, 4] #(96,)

# for문으로 번거롭게 만들지 않아도 된다. but 차이는 y.shape가 (96,) // (96,1) 인 점이다.
# datasets = split_x(dataset, size)

# bbb = []
# ccc = []
# for i in range(len(datasets)):    
#     bbb.append(datasets[i][0:4])
#     ccc.append(datasets[i][4:5])
# x = np.array(bbb) #(96,4)
# y = np.array(ccc) #(96,1)

# x_pred = np.array([97,98,99,100]) #(4,)
# x_pred = x_pred.reshape(1,4) #(1,4)

x = np.reshape(x, (x.shape[0], x.shape[1],1))  #(96,4,1)

x_pred = np.array([97,98,99,100]) #(4,)
x_pred = x_pred.reshape(1,4,1) #(1,4,1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, = train_test_split(x, y, train_size=0.7) 

print(x_train.shape) #차원에 상관없이 train_test_split은 가능하다 비례에 맞게 잘라준다


#2. 모델구성
# model = Sequential()
# model.add(LSTM(100, activation='relu', input_shape=(4,1)))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1))
model = load_model("./save/keras28.h5") #경로에 존재하는 모델 파일을 불러온다.

# model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# lstm (LSTM)                  (None, 100)               40800
# _________________________________________________________________
# dense (Dense)                (None, 50)                5050
# _________________________________________________________________
# dense_1 (Dense)              (None, 10)                510
# _________________________________________________________________
# dense_2 (Dense)              (None, 1)                 11
# =================================================================
# Total params: 46,371
# Trainable params: 46,371
# Non-trainable params: 0
# _________________________________________________________________

model.add(Dense(5 , name ='dense_5')) #기존에 모델에 이어서 추가한 경우, 기존에 생성된 model_name을 제외한 name을 추가하면 기존 load된 model에서 추가가 가능하다
model.add(Dense(1 , name ='dense_6'))
model.summary()

# 에러코드
# Traceback (most recent call last):
#   File "d:\Study\bit_seoul\keras\keras29_load.py", line 52, in <module>
# ValueError: All layers added to a Sequential model should have unique names. 
# Name "dense" is already the name of a layer in this model. Update the `name` argument to pass a unique name.
# dense는 이미 이 모델의 이름으로 존재한다. name을 유니크한 이름으로 업데이트 해라

#Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# lstm (LSTM)                  (None, 100)               40800
# _________________________________________________________________
# dense (Dense)                (None, 50)                5050
# _________________________________________________________________
# dense_1 (Dense)              (None, 10)                510
# _________________________________________________________________
# dense_2 (Dense)              (None, 1)                 11
# _________________________________________________________________
# dense_3 (Dense)              (None, 5)                 10           -> dense_3과 dense_4가 추가된 것을 볼 수 있다.
# _________________________________________________________________
# dense_4 (Dense)              (None, 1)                 6
# =================================================================
# Total params: 46,387
# Trainable params: 46,387
# Non-trainable params: 0
# _________________________________________________________________

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping #EarlyStopping 추가 - 조기종료
early_stopping = EarlyStopping(monitor='loss', patience=150, mode='min') 
model.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=1, callbacks=[early_stopping])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_pred)
print("y_predict : \n", y_predict)

# 결과값
# loss :  0.004994069691747427
# y_predict :
#  [[100.84526]]