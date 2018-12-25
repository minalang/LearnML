import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from zlib import crc32

file_name = 'C:\\Users\\khj\\Desktop\\Hands-On ML files\\housing.csv'
dt = pd.read_csv(file_name)

# 1. 데이터 구조 확인해보기..
#print(dt["ocean_proximity"].value_counts()) # Categories
#print(dt.head())
#dt.info()
#print(dt.describe())
#print(dt.describe(include='all'))


#dt.hist(bins=50, figsize=(20,15))
#plt.show()

# 2. Test Set 만들기..
# def split_train_test(data, test_ratio):
#     shuffled_indices = np.random.permutation(len(data))
#     test_set_size = int(len(data)*test_ratio)
#     test_indices = shuffled_indices[:test_set_size]
#     train_indices = shuffled_indices[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices]

# train_set, test_set = split_train_test(dt, 0.2)
# print(len(train_set), "train + ", len(test_set), "test")

#3. sklearn을 이용한 Test Set 만들기..

# 전체 데이터
# dt["income_cat"] = np.ceil(dt["median_income"] /1.5)
# dt["income_cat"].where(dt["income_cat"] < 5, 5.0, inplace=True)
# print(dt["income_cat"].value_counts() / len(dt))

# # Randomized sampling
# dt["income_cat"] = np.ceil(dt["median_income"] /1.5)
# dt["income_cat"].where(dt["income_cat"] < 5, 5.0, inplace=True)
# train_set1, test_set2 = train_test_split(dt, test_size=0.2, random_state=42)
# print(train_set1["income_cat"].value_counts() / len(train_set1))

# # Stratified sampling
# dt["income_cat"] = np.ceil(dt["median_income"] /1.5)
# dt["income_cat"].where(dt["income_cat"] < 5, 5.0, inplace=True)

# split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# strat_train_set = []
# strat_test_set = []
# for train_index, test_index in split.split(dt, dt["income_cat"]):
#     strat_train_set = dt.loc[train_index]
#     strat_test_set = dt.loc[test_index]
#     print(strat_train_set["income_cat"].value_counts() / len(strat_train_set))

# for set_ in (strat_train_set, strat_test_set): # 임의 생성한 카테고리 삭제
#     set_.drop("income_cat", axis=1, inplace=True)

# strat_train_set.hist(bins=50, figsize=(20,15))
# plt.show()

# 4. 데이터 시각화

#dt.plot(kind="scatter", x="longitude", y="latitude") # 산점도 시각화 x,y평면
#dt.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1) # 데이터가 밀집된 곳 

# dt.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
#     s=dt["population"]/100, label="population", figsize=(10,7),
#     c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True, sharex=False)

# #plt.legend() ?
# plt.show()

# 5. 상관관계 조사

# 모든 특성 간 표준 상관계수;standrad correlation coefficient; 피어슨r
#corr_matrix = dt.corr()
#print(corr_matrix["median_house_value"].sort_values(ascending=False))

# 특성 사이의 r를 산점도로..
# attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
# pd.plotting.scatter_matrix(dt[attributes], figsize=(12,8))
# plt.show()

# 가장 유용해 보이는 것을 다시..
# dt.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
# plt.show()


# 6. 데이터 준비
# 1) 데이터 정제
# 2) 텍스트와 범주형 데이터 변환
# 3) 나만의 변환기 설계
# 4) Feature scaling (min-max, standardization)
# 5) Pipeline

# 7. 모델 선택과 훈련



