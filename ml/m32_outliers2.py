#outlier - 이상치 제거
#outliers1 데이터를 행렬로 데이터를 수정하고 컬럼마다 이상치 제거해보기
#percentile 검색해보기

import numpy as np
def outliers(data_out):     #1/4, 3/4 지점 두개를 뺀 것의 1.5배
    quartile_1, quartile_3 = np.percentile(data_out, [25, 75])     # 25퍼와 75퍼의 지점을 각각 변수에 저장
    print("1사분위 : ", quartile_1) #3.25
    print("3사분위 : ", quartile_3) #97.5
    iqr = quartile_3 - quartile_1 # 94.25
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))

a = np.array([[1,2,3,4,10000,6,7,5000,90,100],                                             
              [10000,20000,3,40000,50000,60000,70000,8,90000,100000]])

a = a.transpose()

print(a)


# outliers(a)
# 전체 데이터 길이의 1/4지점과, 3/4지점을 찾는 것이다
# 1사분위 :  3.25
# 3사분위 :  97.5

b = outliers(a[0][:])
print("이상치 인덱스 : ", b)
print("이상치 : ", a[b])

# 이상치 인덱스 :  (array([4, 7], dtype=int64),)
# 이상치 :  [10000  5000]
