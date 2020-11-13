#데이터를 x = 4개, y = 1개로 나누고 훈련까지!

import numpy as np   

#1. 데이터
dataset = np.array(range(1,11))
size = 5

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset]) 
    # print(type(aaa))
    return np.array(aaa)

datasets = split_x(dataset, size)

bbb = []
ccc = []
for i in range(len(datasets)):    
    bbb.append(datasets[i][0:4])
    ccc.append(datasets[i][4:5])
x = np.array(bbb)
y = np.array(ccc)

print(x)
print(y)

# 결과 x
# [[1 2 3 4]
#  [2 3 4 5]
#  [3 4 5 6]
#  [4 5 6 7]
#  [5 6 7 8]
#  [6 7 8 9]]

# 결과 y
# [[ 5]
#  [ 6]
#  [ 7]
#  [ 8]
#  [ 9]
#  [10]]

'''
#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM 

model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(4,1)))
model.add(Dense(180, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(110, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.summary()
'''

'''
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping #EarlyStopping 추가 - 조기종료
early_stopping = EarlyStopping(monitor='loss', patience=200, mode='min') 
model.fit(x, y, epochs=1000, batch_size=1, verbose=1, callbacks=[early_stopping])


'''