import numpy as np
import pandas as pd

wine = pd.read_csv('./data/csv/winequality-white.csv',
                 header = 0,
                 index_col=None,
                 sep=';')

count_data = wine.groupby('quality')['quality'].count() #quality를 그룹지어놓고 quality안의 데이터들을 묶어서 카운트를 세겠다

# print(count_data)
# quality
# 3      20
# 4     163
# 5    1457
# 6    2198
# 7     880
# 8     175
# 9       5

import matplotlib.pyplot as plt
count_data.plot()
plt.show()

#데이터의 갯수가 다름, 가운데에 몰려있다 - 5,6,7
#개 900만장, 고양이 10장 훈련시키면 거의 개로 생각한다
#데이터의 숫자가 한쪽으로 치우쳐 있게 되면, 그 데이터로 인식할 확률이 높아진다
#거의 5,6,7이라고 판단하고 70프로의 acc를 갖게됨 -  이것이 문제점
#데이터셋 분포가 몰려있기 때문에 y_column을 조절해보자! - 작게 잡아준다


