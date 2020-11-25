import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape) #(442, 10)
print(y.shape) #(442,)

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_) #누적된 합 #pca의 컬럼을 축소했을때의 컬럼의 가치를 누적된 합으로 표시하겠다

print(cumsum)

n_components = np.argmax(cumsum >=0.95) + 1

print(cumsum >= 0.95)
print(n_components)

import matplotlib.pyplot as plt
import seaborn 
plt.plot(cumsum)
plt.grid()
plt.show()