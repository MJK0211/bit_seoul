import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape) #(442, 10)
print(y.shape) #(442,)


# 1.1 데이터 전처리
pca = PCA(n_components=4) #n_components : 줄일 차원의 수
x2d = pca.fit_transform((x)) #fit_transform : 한번에 fit+transform
print(x2d.shape) #(442, 4)

pca_EVR = pca.explained_variance_ratio_ 
print(pca_EVR) #n_components의 개수(4개)만큼 출력 됩니다.
#[0.40242142 0.14923182 0.12059623 0.09554764]
print(sum(pca_EVR))
#n_components=4 일때, 0.767797111020965
