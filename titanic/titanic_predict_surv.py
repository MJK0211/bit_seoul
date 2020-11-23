import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('./data/titanic/csv/train.csv')
test = pd.read_csv('./data/titanic/csv/test.csv')
                
# print(train.shape) #(891, 12)
# print(test.shape) #(418, 11)
# print(train.head())
# print(train.tail())

# Survived - 생존 여부 (0 = 사망, 1 = 생존)
# Pclass - 티켓 클래스 (1 = 1등석, 2 = 2등석, 3 = 3등석)
# Sex - 성별
# Age - 나이
# SibSp - 함께 탑승한 자녀 / 배우자 의 수
# Parch - 함께 탑승한 부모님 / 아이들 의 수
# Ticket - 티켓 번호
# Fare - 탑승 요금
# Cabin - 수하물 번호
# Embarked - 선착장 (C = Cherbourg, Q = Queenstown, S = Southampton)

# train의 칼럼별 결측치 합계
# print(train.isnull().sum())
# PassengerId      0
# Survived         0
# Pclass           0
# Name             0
# Sex              0
# Age            177
# SibSp            0
# Parch            0
# Ticket           0
# Fare             0
# Cabin          687
# Embarked         2

# test의 칼럼별 결측치 합계
# print(test.isnull().sum())
# PassengerId      0
# Pclass           0
# Name             0
# Sex              0
# Age             86
# SibSp            0
# Parch            0
# Ticket           0
# Fare             1
# Cabin          327
# Embarked         0

def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['Survived', 'Dead']
    df.plot(kind='bar', stacked=True, figsize=(15,8))

# bar_chart('Embarked')
#Pclass, Sex, SibSp, Parch, Embarked

train = train.drop(['Cabin', 'Embarked', 'Ticket', 'PassengerId'],axis=1)
test = test.drop(['Cabin', 'Embarked', 'Ticket', 'PassengerId'],axis=1)

# print(test)
train["Age"].fillna(train.groupby("Sex")["Age"].transform("mean"), inplace=True)
test["Age"].fillna(test.groupby("Sex")["Age"].transform("mean"), inplace=True)
test["Fare"].fillna(test.groupby("Sex")["Fare"].transform("median"), inplace=True)


# print(train.isnull().sum())
# print(test.isnull().sum())

train_test_data = [train, test]

for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

sex_mapping = {"male": 0, "female": 1}
title_mapping = {"Mr":0, "Miss":1, "Mrs":2, "Master":3, "Dr":3, "Rev":3, "Col":3, "Major":3, 
                 "Mlle":3, "countess":3, "Ms":3, "Lady":3, "Jonkheer":3, "Don":3, "Dona":3,
                 "Mme":3, "Capt":3, "Sir":3 }
                 
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping) #title 맵핑 '0', '1', '2'
    dataset['Sex'] = dataset['Sex'].map(sex_mapping) #Sex '남:0', '여:1'로 맵핑
    dataset.drop('Name', axis=1, inplace=True) #Name에서 Title 추출 후, 필요없는 데이터이기 때문에 삭제!

# bar_chart('Title')
# plt.show()

# print(test)

#Binning - Age를 10대, 20대, 30대, 40대, 50대, 50대 이상으로 분류
for dataset in train_test_data:
    dataset.loc[dataset['Age'] <= 19, 'Age'] = 0,                             #0 : 10대
    dataset.loc[(dataset['Age'] > 19) & (dataset['Age'] <= 29), 'Age'] = 1,   #1 : 20대
    dataset.loc[(dataset['Age'] > 29) & (dataset['Age'] <= 39), 'Age'] = 2,   #2 : 30대
    dataset.loc[(dataset['Age'] > 39) & (dataset['Age'] <= 49), 'Age'] = 3,   #3 : 40대  
    dataset.loc[(dataset['Age'] > 49) & (dataset['Age'] <= 59), 'Age'] = 4,   #4 : 50대  
    dataset.loc[dataset['Age'] > 59, 'Age'] = 5,                              #5 : 50대이상

# print(train.groupby('Survived')['Age'].value_counts())
# bar_chart('Age')
# plt.show()


