#keras30_save.py 에서 구성한 모델을
#keras31_load.py에서 저장한 파일을 불러오고 input_shape를 바꿀 방법을 찾아봐라 input_shape(3,1) -> (4,1)로 변경
#Sequencial로만 생각하지 말고 함수형으로 해결하면 될 듯?
#과제 완성시키기 11/13

import numpy as np  
from tensorflow.keras.models import Model, load_model #load_model 추가!
from tensorflow.keras.layers import Dense, LSTM, Input, InputLayer, Layer

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

x = np.reshape(x, (x.shape[0], x.shape[1],1))  #(96,4,1)

x_pred = np.array([97,98,99,100]) #(4,)
x_pred = x_pred.reshape(1,4,1) #(1,4,1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, = train_test_split(x, y, train_size=0.7) 

#2. 모델구성

model = load_model("./save/keras28_1.h5")
model.summary()

# Model: "functional_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         [(None, 3, 1)]            0
# _________________________________________________________________
# lstm_layer (LSTM)            (None, 200)               161600
# _________________________________________________________________
# dense1 (Dense)               (None, 180)               36180
# _________________________________________________________________
# dense2 (Dense)               (None, 150)               27150
# _________________________________________________________________
# dense3 (Dense)               (None, 110)               16610
# _________________________________________________________________
# dense4 (Dense)               (None, 60)                6660
# _________________________________________________________________
# dense5 (Dense)               (None, 10)                610
# _________________________________________________________________
# output1 (Dense)              (None, 1)                 11
# =================================================================
# Total params: 248,821
# Trainable params: 248,821
# Non-trainable params: 0
# _________________________________________________________________

input1 = Input(shape=(4, 1))
dense = model(input1)
output1 = Dense(1)(dense)
model = Model(inputs=input1, outputs=output1)
model.summary()

# Model: "functional_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         [(None, 4, 1)]            0             -> Input Layer shape 변경
# _________________________________________________________________
# functional_1 (Functional)    (None, 1)                 248821        -> 저장된 모델의 parameter 총합이 나온다
# _________________________________________________________________
# dense (Dense)                (None, 1)                 2
# =================================================================
# Total params: 248,823
# Trainable params: 248,823
# Non-trainable params: 0
# _________________________________________________________________

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping 
early_stopping = EarlyStopping(monitor='loss', patience=150, mode='min') 
model.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=1, callbacks=[early_stopping])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_pred)
print("y_predict : \n", y_predict)

#결과값

# loss :  0.001984312664717436
# y_predict :
#  [[100.90826]]
