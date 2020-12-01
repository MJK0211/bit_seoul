import numpy as np

nc = np.load('./project/data/npy/nc.npy', allow_pickle=True)
wg = np.load('./project/data/npy/wage.npy', allow_pickle=True)
sn = np.load('./project/data/npy/snack.npy', allow_pickle=True)

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
count_list = np.array(count_list)

wg = np.delete(wg, 0, axis=1)
sn = np.delete(sn, 0, axis=1)
result = np.array([])
snack = np.array([])
for i in range(len(wg)):
    result = np.append(np.tile(wg[i], count_list[i]), result)
    snack = np.append(np.tile(sn[i], count_list[i]), snack)

print(result)

result = np.sort(result)
snack = np.sort(snack)
print(snack)
print(result)


np.save('./project/data/npy/wage_result.npy', arr=result)
np.save('./project/data/npy/snack_result.npy', arr=snack)

