#차원을 축소했다가 증폭했다는 개념
#pca - 1. 차원축소, 2. 특성추출 - 즉, 중요한 부분빼고는 지워진다
#즉 오토인코더를 사용하면 차원축소, 특성추출이 가능하다
#오토인코더는 미백, 채색 같은 잡티제거 기능으로 많이 사용된다

import numpy as np
from tensorflow.keras.datasets import mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, _ ), (x_test, _ ) = mnist.load_data()

x_train = x_train.reshape(60000,784).astype('float32')/255.
x_test = x_test.reshape(10000,784).astype('float32')/255.

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,), activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

model_01 = autoencoder(hidden_layer_size=1)
model_02 = autoencoder(hidden_layer_size=2)
model_04 = autoencoder(hidden_layer_size=4)
model_08 = autoencoder(hidden_layer_size=8)
model_16 = autoencoder(hidden_layer_size=16)
model_32 = autoencoder(hidden_layer_size=32)

model_01.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model_01.fit(x_train, x_train, epochs=10)

model_02.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model_02.fit(x_train, x_train, epochs=10)

model_04.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model_04.fit(x_train, x_train, epochs=10)

model_08.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model_08.fit(x_train, x_train, epochs=10)

model_16.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model_16.fit(x_train, x_train, epochs=10)

model_32.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model_32.fit(x_train, x_train, epochs=10)


output_01 = model_01.predict(x_test)
output_02 = model_02.predict(x_test)
output_04 = model_04.predict(x_test)
output_08 = model_08.predict(x_test)
output_16 = model_16.predict(x_test)
output_32 = model_32.predict(x_test)

from matplotlib import pyplot as plt
import random

fig, axes = plt.subplots(7, 5, figsize=(15,15))

random_imgs = random.sample(range(output_01.shape[0]), 5)
outputs = [x_test, output_01, output_02, output_04, output_08, output_16, output_32]

for row_num, row in enumerate(axes):
    for col_num, ax in enumerate(row):
        ax.imshow(outputs[row_num][random_imgs[col_num]].reshape(28,28), cmap='gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
plt.show()