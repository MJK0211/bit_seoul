import numpy as np

nc = np.load('./project/data/npy/nc.npy', allow_pickle=True)
am = np.load('./project/data/npy/america.npy', allow_pickle=True)
we = np.load('./project/data/npy/weather.npy', allow_pickle=True)
wg = np.load('./project/data/npy/wage.npy', allow_pickle=True)
sn = np.load('./project/data/npy/snack.npy', allow_pickle=True)

# print(nc.shape) #(2275, 5)
# print(am.shape) #(2275, 5)
# print(we.shape) #(2275, 4)
# print(wg.shape) #(10, 2)
# print(sn.shape) #(10, 2)                                   

# print(nc[-1]) #['2020-11-23' 822000.0 831000.0 836000.0 819000.0] - "날짜","종가","오픈","고가","저가"
# print(am[-1]) #['2020-11-23' 1113.51 1115.08 1116.82 1109.91] - "날짜","종가","오픈","고가","저가",
# print(we[-1]) #['2020-11-23' 1.7 -1.5 5.9] -날짜, 평균기온(℃), 최저기온(℃), 최고기온(℃)


year = list()
for i in range(len(wg)):
    year.append(str(wg[i,0]))

count_list = list()    

for j in range(len(year)):
    count = 0
    for i in range(len(nc)):     
        a = np.char.count(nc[i,0], year[j])
        count += a        
    count_list.append(count)
    
print(count_list)

# print(nc[:,0])

# 88
# 248
# 245
# 245
# 248
# 246
# 243
# 244
# 246
# 222