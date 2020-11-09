import numpy as np 

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

from tensorflow.keras.models import Sequential #텐서플로안에 케라스안에 모델스안에 시퀀셜(순차적)을 가져오겠다
from tensorflow.keras.layers import Dense #layer는 딥러닝의 단계(층)

#2. 모델구성
model = Sequential() #방향이 순차적인 모델이다
model.add(Dense(3, input_dim=1)) #3개의 층을 쌓고 input_dimension 인풋하는 차원을 한개로 하겠다. y=wx+b // Dense = 단순 DNN계층
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1)) #계층간 hyper parameter tuning?


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#loss는 선과 데이터의 차이값이므로 최적의 loss값은 0이다. MSE(Mean Squared Error)는 손실함수이다. 우리는 손실을 최소화 하기위해 'mse'를 사용하고, optimizer(최적화)는 'adams'를 쓰겠다. acc'정확성

model.fit(x, y, epochs=200, batch_size=2) #우리는 정제된 데이터로 이 모델을 훈련시키겠다. 'epochs' 100번 훈련시키겠다. 'batch_size' 한개씩 넣겠다.

#4. 평가, 예측
loss, acc = model.evaluate(x, y, batch_size=2) #평가, 예측한 값을 반환시키기 위해

y = model.predict(x)
print("loss: ", loss)
print("acc: ", acc)
print(y)
#과제 acc는 왜 0.2인가? 1.0으로 맞춰라!

#앞으로 이력서에 남기는 것은 hyperparameter  tuning 이다
