# 넘파이 불러와서
# .fit 으로 코딩

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
x_train = np.load('./data/npy/keras63_train_x.npy', allow_pickle=True)
y_train = np.load('./data/npy/keras63_train_y.npy', allow_pickle=True)
x_test = np.load('./data/npy/keras63_test_x.npy', allow_pickle=True)
y_test = np.load('./data/npy/keras63_test_y.npy', allow_pickle=True)

#2. 모델 구성
from tensorflow.python.keras.utils.data_utils import Sequence
import autokeras as ak

clf = ak.ImageClassifier(
    overwrite=True,
    max_trials=2
)

#3. 컴파일, 훈련
clf.fit(x_train, y_train, epochs=50)
  
#4. 평가, 예측
loss = clf.evaluate(x_test, y_test)
print("loss : ", loss)

y_pred = clf.predict(x_test)
print("y_pred : ", y_pred)

