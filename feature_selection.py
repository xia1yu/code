import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt


# Step1 读取数据
data = pd.read_excel("./data.xlsx")
# data = np.array(data)
data_x = data.iloc[1:50001,0:-1]
y = data.iloc[1:50001,-1:]


# 在选择的数据中，选择70%作为训练集，30%作为测试集
X_train, X_test, y_train, y_test = train_test_split(data_x, y, test_size=0.3, random_state=1036, shuffle = True)

# 归一化
min_max_scaler = preprocessing.MinMaxScaler()
X_train_scaled = min_max_scaler.fit_transform(X_train)
X_test_scaled = min_max_scaler.fit_transform(X_test)


# # 特征选择
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
svc = SVC(kernel="linear")
dt = DecisionTreeClassifier()
rfecv = RFECV(estimator=dt, step=1, cv=StratifiedKFold(2), scoring='accuracy')
rfecv.fit(X_train, y_train)
print("Optimal number of features : %d" % rfecv.n_features_)
print("Ranking of features names: %s" % X_train.columns[rfecv.ranking_-1])
print("Ranking of features nums: %s" % rfecv.ranking_)
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.savefig("feature.jpg")
plt.show()

# 特征对比图
import seaborn as sns
sns.pairplot(X_train, vars=["wind_speed","generator_speed", "power"],
             palette="husl"
            ,diag_kind="kde")
plt.savefig("duibi.jpg")