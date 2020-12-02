from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

#2. 모델구성
model = Sequential() 
model.add(Dense(3, input_dim=1)) 
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1)) 

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

optimizer = [Adam(lr=0.001), Adadelta(lr=0.001), Adamax(lr=0.001), Adagrad(lr=0.001), RMSprop(lr=0.001), SGD(lr=0.001), Nadam(lr=0.001)]
# optimizer = Adam(lr=0.001) 
# optimizer = Adadelta(lr=0.001) 
# optimizer = Adamax(lr=0.001) 
# optimizer = Adagrad(lr=0.001) 
# optimizer = RMSprop(lr=0.001) 
# optimizer = SGD(lr=0.001) 
# optimizer = Nadam(lr=0.001) 

loss = list()
y_pred = list()
for i in range(len(optimizer)): 
    model.compile(loss='mse', optimizer=optimizer[i], metrics=['mse'])
    model.fit(x, y, epochs=100, batch_size=1)
    loss.append(model.evaluate(x, y, batch_size=1))   
    y_pred.append(model.predict([11]))

for i in range(len(optimizer)):
    print("optimizer : ", optimizer[i]._name)
    print("loss : ", loss[i], "result : ", y_pred[i])
 

# 결과값
# optimizer :  <tensorflow.python.keras.optimizer_v2.adam.Adam object at 0x000002AD94BE0280>
# loss :  [0.00021147615916561335, 0.00021147615916561335] result :  [[10.983959]]

# optimizer :  <tensorflow.python.keras.optimizer_v2.adadelta.Adadelta object at 0x000002AD94BE0340>
# loss :  [0.00015014977543614805, 0.00015014977543614805] result :  [[10.984099]]

# optimizer :  <tensorflow.python.keras.optimizer_v2.adamax.Adamax object at 0x000002AD94BE0EE0>
# loss :  [3.765876499528531e-13, 3.765876499528531e-13] result :  [[10.999999]]

# optimizer :  <tensorflow.python.keras.optimizer_v2.adagrad.Adagrad object at 0x000002AD94BE0E20>
# loss :  [6.423306439691523e-13, 6.423306439691523e-13] result :  [[11.]]

# optimizer :  <tensorflow.python.keras.optimizer_v2.rmsprop.RMSprop object at 0x000002AD94BE94F0>
# loss :  [0.001034580753184855, 0.001034580753184855] result :  [[10.943472]]

# optimizer :  <tensorflow.python.keras.optimizer_v2.gradient_descent.SGD object at 0x000002AD94BE9340>
# loss :  [8.949004870473232e-10, 8.949004870473232e-10] result :  [[11.000055]]

# optimizer :  <tensorflow.python.keras.optimizer_v2.nadam.Nadam object at 0x000002AD94BE9730>
# loss :  [1.0161675163544714e-05, 1.0161675163544714e-05] result :  [[10.994553]]
