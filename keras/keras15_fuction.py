#1. 데이터
import numpy as np
x = np.array([range(1,101), range(711,811), range(100)])
y = np.array(range(101,201))

x = np.transpose(x)

from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

print(x)
print(x.shape) 

#2. 모델구성
from tensorflow.keras.models import Sequential, Model #함수형 모델 사용 
from tensorflow.keras.layers import Dense, Input #함수형 모델은 input layer가 별도로 있다

# model = Sequential()
# model.add(Dense(5, input_shape=(3,), activation='relu'))
# model.add(Dense(4, activation='relu'))
# model.add(Dense(3, activation='relu'))
# model.add(Dense(1)) 

# model.summary()
# 결과값
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #     -> Output Shape의 차원은 현재 레이어에 있는 노드의 수
# =================================================================  -> Param  (현재 노드의 수 + Bias) * (input의 차원 or 이전의 차원)
# dense (Dense)                (None, 5)                 20          
# _________________________________________________________________  -> y=wx+b bias라는 놈을 생각을 안함, 단순 곱이라고 생각하면 3(input)* 5(현재layer 노드의 수), but bias가 존재한다
# dense_1 (Dense)              (None, 4)                 24          -> 모든 layer마다 bias가 준비가 되어있다.
# _________________________________________________________________  -> input이 3이지만 bias를 하나로 잡아주고 param을 계산하면 (3+1)*5 = 20이된다.
# dense_2 (Dense)              (None, 3)                 15          -> Total params 는 Param의 총합
# _________________________________________________________________
# dense_3 (Dense)              (None, 1)                 4
# =================================================================
# Total params: 63
# Trainable params: 63
# Non-trainable params: 0
# _________________________________________________________________

input1 = Input(shape=(3,)) #input layer 구성
dense1 = Dense(5, activation='relu')(input1) #첫번째 dense층 구성, 상단에 input layer를 사용하겠다.
#activation = 활성화 함수, layer마다 활성화 함수가 있다. 통상 relu를 사용하면 평균 85점이상 나온다 
#회귀 = regress, 선형 = linear
#선형회귀 모델에서 activation default는 'linear'를 사용하고 있다 (activation='linear')
dense2 = Dense(4, activation='relu')(dense1) 
dense3 = Dense(3, activation='relu')(dense2)
output1 = Dense(1)(dense3) #마지막 activation은 linear이여야 하기 때문에 생략함
model = Model(inputs = input1, outputs = output1) #모델을 정의함, 어디서부터 어디까지 모델구성인지

model.summary()
# 결과값
# Model: "functional_1"                                              -> 함수형 모델
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         [(None, 3)]               0           -> Sequencial layer와 차이는 input layer가 추가된 점이다
# _________________________________________________________________
# dense (Dense)                (None, 5)                 20
# _________________________________________________________________
# dense_1 (Dense)              (None, 4)                 24
# _________________________________________________________________
# dense_2 (Dense)              (None, 3)                 15
# _________________________________________________________________
# dense_3 (Dense)              (None, 1)                 4
# =================================================================
# Total params: 63
# Trainable params: 63
# Non-trainable params: 0
# _________________________________________________________________


'''
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1, validation_split=0.25, verbose=0)

#4. 평가, 예측
loss = model.evaluate(x_train, y_train, batch_size=1)
y_pred = model.predict(x_test)

print("y_test : \n", y_test)
print("y_pred : \n", y_pred)

from sklearn.metrics import mean_absolute_error
def RMSE(y_test, y_pred):
    return np.sqrt(mean_absolute_error(y_test, y_pred))
print("RMSE : ", RMSE(y_test, y_pred))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R2 : ", r2)
'''