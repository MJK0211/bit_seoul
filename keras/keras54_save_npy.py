from sklearn.datasets import load_iris 
import numpy as np

iris = load_iris()
print(iris)
print(type(iris)) # <class 'sklearn.utils.Bunch'>

x_data = iris.data
y_data = iris.target


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8) 

np.save('./data/iris_x_train.npy', arr=x_train)
np.save('./data/iris_x_test.npy', arr=x_test)
np.save('./data/iris_y_train.npy', arr=y_train)
np.save('./data/iris_y_test.npy', arr=y_test)
