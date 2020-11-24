#분류, 클래스파이어 모델들 추출

import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
import warnings

warnings.filterwarnings('ignore')

iris = pd.read_csv('./data/csv/iris_ys.csv', header=0)
x = iris.iloc[:, :4]
y = iris.iloc[:, -1] #(150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66, shuffle=True)

allAlgorithms = all_estimators(type_filter='classifier') #클래스파이어 모델들을 추출

for(name, algorithms) in allAlgorithms:
    try:
        kfold = KFold(n_splits=5, shuffle=True) #5개로 조각을 내고 섞어서 하겠다
      
        model = algorithms()
        model.fit(x_train, y_train)
        scores = cross_val_score(model, x_train, y_train, cv=kfold) #검증된 scores를 모델과 엮어줌
        print(model, '의 정답률: ', scores)
    except:
        pass

# AdaBoostClassifier() 의 정답률:  [1.         1.         1.         1.         0.95833333]
# BaggingClassifier() 의 정답률:  [1. 1. 1. 1. 1.]
# BernoulliNB() 의 정답률:  [0.33333333 0.20833333 0.20833333 0.16666667 0.33333333]
# CalibratedClassifierCV() 의 정답률:  [0.75       0.875      0.91666667 0.75       0.95833333]
# ComplementNB() 의 정답률:  [0.54166667 0.66666667 0.625      0.70833333 0.79166667]
# DecisionTreeClassifier() 의 정답률:  [1.         1.         1.         1.         0.95833333]
# DummyClassifier() 의 정답률:  [0.25       0.5        0.29166667 0.16666667 0.29166667]       
# ExtraTreeClassifier() 의 정답률:  [1.         1.         0.91666667 0.91666667 0.95833333]
# ExtraTreesClassifier() 의 정답률:  [1.         1.         1.         0.95833333 1.        ]
# GaussianNB() 의 정답률:  [1.         1.         0.95833333 1.         0.95833333]
# GaussianProcessClassifier() 의 정답률:  [1. 1. 1. 1. 1.]
# GradientBoostingClassifier() 의 정답률:  [1.         1.         1.         1.         0.95833333]
# HistGradientBoostingClassifier() 의 정답률:  [1.         0.95833333 1.         0.95833333 1.        ]
# KNeighborsClassifier() 의 정답률:  [1.         0.95833333 1.         1.         0.95833333]
# LabelPropagation() 의 정답률:  [1. 1. 1. 1. 1.]
# LabelSpreading() 의 정답률:  [1. 1. 1. 1. 1.]
# LinearDiscriminantAnalysis() 의 정답률:  [0.95833333 1.         1.         1.         1.        ]
# LinearSVC() 의 정답률:  [0.625      0.75       0.66666667 0.91666667 0.875     ]
# LogisticRegression() 의 정답률:  [1. 1. 1. 1. 1.]

