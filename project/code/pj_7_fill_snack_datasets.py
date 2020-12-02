import numpy as np

nc = np.load('./project/data/npy/nc.npy', allow_pickle=True)
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

print(count_list) #[ 88 248 245 245 248 246 243 244 246 222]

sn = np.delete(sn, 0, axis=1)

snack = np.array([])
for i in range(len(sn)):   
    snack = np.append(snack, np.tile(sn[i], count_list[i]))  
print(snack)

np.save('./project/data/npy/snack_result.npy', arr=snack)
np.save('./project/data/npy/count_list.npy', arr=count_list)