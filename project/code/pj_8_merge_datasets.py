import numpy as np

nc = np.load('./project/data/npy/nc.npy', allow_pickle=True) #(2275,4)
am = np.load('./project/data/npy/america.npy', allow_pickle=True) #(2275,4)
weather = np.load('./project/data/npy/weather.npy', allow_pickle=True) #(2275,3)
snack_result = np.load('./project/data/npy/snack_result.npy', allow_pickle=True).astype('float32') #(2275,)
wage_result = np.load('./project/data/npy/wage_result.npy', allow_pickle=True).astype('float32') #(2275,)

nc = np.delete(nc, 0, axis=1).astype('float32')
am = np.delete(am, 0, axis=1).astype('float32')
weather = np.delete(weather, 0, axis=1).astype('float32')
snack_result = snack_result.reshape(2275,1)

data_all = np.concatenate((nc, am, weather, snack_result), axis=1)

np.save('./project/data/npy/data_all.npy', arr=data_all)

