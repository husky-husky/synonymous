# coding: gbk
# @Author    :陈梦淇
# @time      :2022/10/11
import pandas as pd
import numpy as np

from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
import lightgbm as lgb

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix

train = pd.read_excel("final_data1001_add_vex.xlsx")
features = train.iloc[:, 5: -3]
labels = train.iloc[:, 0]
columns = features.columns.tolist()

kf = KFold(n_splits=5, random_state=2021, shuffle=True)

clf = lgb.sklearn.LGBMClassifier()
rfe_cv = RFECV(clf, step=1, cv=kf)
rfe_cv = rfe_cv.fit(features, labels)

indx_n = 0
f = open("rfecv_results.txt", "w")
op_feature_list = []
for i in range(0, len(rfe_cv.ranking_)):
    if rfe_cv.ranking_[i] == 1:
        op_feature_list.append(features.columns[i])
        f.write(features.columns[i] + "\n")

f.close()

# 最终选择的特征数量
print("最优特征数 : %d" % rfe_cv.n_features_)
print('最优特征为：', op_feature_list)
# True表示保留特征，False表示剔除特征
print('被选择的特征为Ture :', rfe_cv.support_)
print('特征重要性排序 = :', rfe_cv.ranking_)  # 数值越小越重要
print('5.交叉验证得分随特征数量的变化：', rfe_cv.grid_scores_)
