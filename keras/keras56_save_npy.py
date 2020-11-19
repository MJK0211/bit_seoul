#save dataset 6개 저장

from sklearn.datasets import load_diabetes, load_breast_cancer, load_boston
from tensorflow.keras.datasets import fashion_mnist, cifar10, cifar100
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
np.save('./data/fashion_mnist_x_train.npy', arr=x_train)
np.save('./data/fashion_mnist_x_test.npy', arr=x_test)
np.save('./data/fashion_mnist_y_train.npy', arr=y_train)
np.save('./data/fashion_mnist_y_test.npy', arr=y_test)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
np.save('./data/cifar10_x_train.npy', arr=x_train)
np.save('./data/cifar10_x_test.npy', arr=x_test)
np.save('./data/cifar10_y_train.npy', arr=y_train)
np.save('./data/cifar10_y_test.npy', arr=y_test)

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
np.save('./data/cifar100_x_train.npy', arr=x_train)
np.save('./data/cifar100_x_test.npy', arr=x_test)
np.save('./data/cifar100_y_train.npy', arr=y_train)
np.save('./data/cifar100_y_test.npy', arr=y_test)

from sklearn.model_selection import train_test_split

boston = load_boston()
x = boston.data
y = boston.target
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8) 
np.save('./data/boston_x_train.npy', arr=x_train)
np.save('./data/boston_x_test.npy', arr=x_test)
np.save('./data/boston_y_train.npy', arr=y_train)
np.save('./data/boston_y_test.npy', arr=y_test)

diabetes = load_diabetes()
x = diabetes.data
y = diabetes.target
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8) 
np.save('./data/diabetes_x_train.npy', arr=x_train)
np.save('./data/diabetes_x_test.npy', arr=x_test)
np.save('./data/diabetes_y_train.npy', arr=y_train)
np.save('./data/diabetes_y_test.npy', arr=y_test)

cancer = load_breast_cancer()
x = cancer.data
y = cancer.target
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8) 
np.save('./data/cancer_x_train.npy', arr=x_train)
np.save('./data/cancer_x_test.npy', arr=x_test)
np.save('./data/cancer_y_train.npy', arr=y_train)
np.save('./data/cancer_y_test.npy', arr=y_test)
