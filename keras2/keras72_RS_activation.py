# 70 카피
# activation 하고 완성하기

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Dropout
from tensorflow.keras.datasets import mnist #dataset인 mnist추가
from tensorflow.keras.activations import relu, selu, elu
from tensorflow.keras.layers import Activation 
from tensorflow.keras.layers import ReLU, ELU, LeakyReLU

#1. 데이터
#mnist를 통해 손글씨 1~9까지의 데이터를 사용
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, x_test.shape) #(60000,28,28), (10000,28,28)
# print(y_train.shape, y_test.shape) #(60000,), (10000,)

x_predict = x_test[:10]
x_test = x_test[10:] #(9990,28,28)
y_real = y_test[:10] #(10,)
y_test = y_test[10:] #(9990,)

#1_1. 데이터 전처리 - OneHotEncoding
from tensorflow.keras.utils import to_categorical 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape) #(60000, 10)
print(y_train[0]) #5 - [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.] - 0/1/2/3/4/5/6/7/8/9 - 5의 위치에 1이 표시됨

x_train = x_train.reshape(60000,28*28).astype('float32')/255. 
x_test = x_test.reshape(9990,28*28).astype('float32')/255.
x_predict = x_predict.reshape(10,28*28).astype('float32')/255

#2. 모델
def build_model(drop=0.5, optimizer='adam', lr=0.0001, activation='relu'):
    inputs = Input(shape=(28*28,), name='input')
    x = Dense(12, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(12, activation=activation, name='hidden2')(x)
    # x = Activation('relu')(x)
    # x = LeakyReLU(alpha=0.3)(x) # 대문자 activation는 클래스가 가지고있는 parameter를 수정가능하다 
    # linear는 그값 그대로 던져주는 것, 그렇게 때문에 다음에 Activation을 사용하면 적용되서 다음 레이어로 값을 던져준다
    # 필수적으로 model을 구성할때 activation을 사용하는 것이 아니다. 판단하에 결정하면 된다.
    x = Dropout(drop)(x)
    x = Dense(12, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer(lr=lr), metrics=['acc'], loss='categorical_crossentropy')

    return model

from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

lr = [0.001, 0.01]   
leakyrelu = lambda x: relu(x, alpha=0.1)
def create_hyperparameter(lr):
    # batches = [10, 20, 30, 40, 50]
    batches = [10, 20]
    optimizers = [Adam]  
    dropout = [0.1, 0.5]
    activation = [relu, leakyrelu] # 'leakyrelu'는 에러남, so -> 설정 leakyrelu = lambda x: relu(x, alpha=0.1)
    lr=[0.1, 0.01]
    return {"batch_size" : batches, "optimizer" : optimizers, "drop" : dropout, "lr" : lr, "activation" : activation }

hyperparameters = create_hyperparameter(lr)

print(hyperparameters)

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier #keras를 sklearn으로 쌓겠다. 
model = KerasClassifier(build_fn=build_model, verbose=1) #케라스 모델을 맵핑

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# search = GridSearchCV(model, hyperparameters, cv=3)
search = RandomizedSearchCV(model, hyperparameters, cv=3)

search.fit(x_train, y_train, verbose=1)

print(search.best_params_)
acc = search.score(x_test, y_test)
print("acc : ", acc)

# {'optimizer': <class 'tensorflow.python.keras.optimizer_v2.adam.Adam'>, 'lr': 0.01, 'drop': 0.1, 'batch_size': 20, 'activation': <function <lambda> at 0x00000172A26559D0>}
# 500/500 [==============================] - 1s 1ms/step - loss: 0.3394 - acc: 0.9008
# acc :  0.9008008241653442