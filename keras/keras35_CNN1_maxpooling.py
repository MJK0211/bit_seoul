#Maxpooling & Flatten

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten #MaxPooling2D 추가, Flatten 추가

#1. 데이터

#2. 모델구성
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(10,10,1))) #(9,9,10)
model.add(Conv2D(5, (2,2), padding='same')) #(9,9,5)
model.add(Conv2D(3, (3,3), padding='valid')) #(7,7,3)
model.add(Conv2D(7, (2,2))) #(6,6,7)
model.add(MaxPooling2D()) #기본 Default는 2이다 - (3,3,7)
model.add(Flatten()) #현재까지 내려왔던 것을 일자로 펴주는 기능 - 이차원으로 변경 (3*3*7 = 63,) = (63,) 다음 Dense층과 연결시키기 위해 사용
model.add(Dense(1))
model.summary()

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #     -> (input값 * 커널사이즈 + bias) * 다음 출력 갯수
# =================================================================
# conv2d (Conv2D)              (None, 9, 9, 10)          50          -> (2*2)커널 * 1(흑백채널) * 1(장)(최초입력) * 10(장) + 10(bias) = 4*1*10 + 10
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 9, 9, 5)           205         -> (2*2)커널 * 1(흑백채널) * 10(장)(conv2d_입력) * 5(장)(출력)  + 5(bias) = 4*1*10*5 + 5
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 7, 7, 3)           138         -> (3*3)커널 * 1(흑백채널) * 5(장)(conv2d_1_입력) * 3(장)(출력)  + 3(bias) = 9*1*5*3 + 3
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 6, 6, 7)           91          -> (2*2)커널 * 1(흑백채널) * 3(장)(conv2d_2_입력) * 7(장)(출력)  + 7(bias) = 4*1*3*7 + 7 
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 3, 3, 7)           0           -> Default 2 (2개씩 나눔)
# _________________________________________________________________
# flatten (Flatten)            (None, 63)                0           -> 이차원으로 변경 (3*3*7 = 63) - (63,)
# _________________________________________________________________
# dense (Dense)                (None, 1)                 64          -> (63+1)*1
# =================================================================
# Total params: 548
# Trainable params: 548
# Non-trainable params: 0
# _________________________________________________________________

#3. 컴파일, 훈련






#4. 평가, 예측