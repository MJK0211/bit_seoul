import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Dropout
from tensorflow.keras.datasets import mnist #dataset인 mnist추가

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

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255. 
x_test = x_test.reshape(9990,28,28,1).astype('float32')/255.
x_predict = x_predict.reshape(10,28,28,1).astype('float32')/255

#2. 모델
def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(28,28,1), name='input')
    x = Conv2D(20, kernel_size=(2,2), padding='same')(inputs)
    x = Dropout(drop)(x)
    x = Conv2D(10, kernel_size=(2,2), padding='same')(x)  
    x = Dropout(drop)(x)
    x = MaxPooling2D()(x)
    x = Dropout(drop)(x)
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')

    return model

def create_hyperparameter():
    batches = [10, 20, 30, 40, 50]  
    optimizers = ['rmsprop', 'adam', 'adadelta'] #learning_rate 검색하기
    dropout = [0.1, 0.5, 5]
    return{"batch_size" : batches, "optimizer" : optimizers, "drop" : dropout}

hyperparameters = create_hyperparameter()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier #keras를 sklearn으로 쌓겠다. 
model = KerasClassifier(build_fn=build_model, verbose=1) #케라스 모델을 맵핑

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# search = GridSearchCV(model, hyperparameters, cv=3)
search = RandomizedSearchCV(model, hyperparameters, cv=3)

search.fit(x_train, y_train, verbose=1)

print(search.best_params_)
acc = search.score(x_test, y_test)
print("acc : ", acc)