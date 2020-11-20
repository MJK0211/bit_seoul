import numpy as np
import pandas as pd

kospi200 = np.load('./data/npy/kospi200.npy', allow_pickle=True)
samsung = np.load('./data/npy/samsung.npy', allow_pickle=True)
# print(kospi200)
# print(kospi200.shape) #(426, 5)
# print(samsung) 
# print(samsung.shape) #(426, 5)
 
def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column

        if y_end_number < len(dataset):            
            tmp_x = dataset[i:x_end_number, :]
            tmp_y = dataset[x_end_number:y_end_number, 3]
            x.append(tmp_x)
            y.append(tmp_y)         
        else:
            break

        return np.array(x), np.array(y)
samsung = np.transpose(samsung)
x, y = split_xy5(samsung, 5, 1)

print(x)