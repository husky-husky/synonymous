# coding: gbk
# @Author    :陈梦淇
# @time      :2022/10/10
import pandas as pd

import numpy as np
import time

from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB

from sklearn.feature_selection import RFE

from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix

from train_and_test.blind_test import balance_1


def blind_test(features=None):
    test_data = pd.read_excel("../train_and_test/test-new-7-27.xlsx")
    train_data = pd.read_excel("final_data1001_add_vex.xlsx")
    a = train_data["Class"].value_counts()
    if not features:  # 未指定使用的特征，采用全部特征
        train_features = train_data.iloc[:, 5: -3]
    else:
        train_features = train_data[features]

    train_labels = train_data.iloc[:, 0]

    columns = train_features.columns.tolist()
    test_features = test_data[columns]
    test_features = test_features.iloc[:, :]
    test_labels = test_data.iloc[:, 0]

    # model = xgb.XGBClassifier(num_class=2, max_depth=30, learning_rate=0.1, n_estimators=160, silent=True, objective='multi:softmax')
    # model = RandomForestClassifier(random_state=512145)
    model = lgb.sklearn.LGBMClassifier()
    # model = GradientBoostingClassifier()
    model.fit(train_features, train_labels)

    y_pred = model.predict(test_features)
    balance_1(test_labels, y_pred)

    ACC = []
    F1 = []
    PPV = []
    NPV = []
    TPR = []
    TNR = []
    AUC = []
    MCC = []
    OPM = []
    TP = []
    TN = []
    FP = []
    FN = []
    tn, fp, fn, tp = confusion_matrix(test_labels, y_pred).ravel()
    TP.append(tp)
    FP.append(fp)
    TN.append(tn)
    FN.append(fn)

    acc = accuracy_score(test_labels, y_pred)  # acc
    ACC.append(acc)
    f1 = f1_score(test_labels, y_pred)  # f1
    F1.append(f1)
    ppv = tp / (tp + fp)  # positivate predictive value
    PPV.append(ppv)
    npv = tn / (tn + fn)  # negative predictive value
    NPV.append(npv)
    tpr = tp / (tp + fn)  # sensitivity / true positive rate
    TPR.append(tpr)
    tnr = tn / (tn + fp)  # specificity / true negative rate
    TNR.append(tnr)
    auc = roc_auc_score(test_labels, model.predict_proba(test_features)[:, 1])  # auc
    AUC.append(auc)
    mcc = matthews_corrcoef(test_labels, y_pred)  # mcc Matthews correlation coefficient
    MCC.append(mcc)
    opm = (ppv + npv) * (tpr + tnr) * (acc + (1 + mcc) / 2) / 8  # opm
    OPM.append(opm)
    # return acc

    print('TP:', tp)
    print('TN:', tn)
    print('FP:', fp)
    print('FN:', fn)
    print("svm-测试集ACC：", acc)
    print("svm-测试集F1：", f1)
    print("svm-测试集PPV：", ppv)
    print("svm-测试集NPV：", npv)
    print("svm-测试集TPR：", tpr)
    print("svm-测试集TNR：", tnr)
    print("svm-测试集AUC：", auc)
    print("svm-测试集MCC：", mcc)
    print("svm-测试集OPM：", opm)
    print('-----------------------------------------')
    print(' ')


def train_5cv(features=None):
    train_data = pd.read_excel("final_data1001_add_vex.xlsx")
    if not features:  # 未指定使用的特征，采用全部特征
        train_features_all = train_data.iloc[:, 5: -3]
    else:
        train_features_all = train_data[features]

    train_labels_all = train_data.iloc[:, 0]

    kf = KFold(n_splits=5, random_state=2021, shuffle=True)

    ACC = []
    F1 = []
    PPV = []
    NPV = []
    TPR = []
    TNR = []
    AUC = []
    MCC = []
    OPM = []
    TP = []
    TN = []
    FP = []
    FN = []

    for train_index, test_index in kf.split(train_features_all):
        train_features, test_features = train_features_all.iloc[train_index, :], train_features_all.iloc[test_index, :]  # 此处需要加上iloc
        train_labels, test_labels = train_labels_all[train_index], train_labels_all[test_index]

        # lightGBM
        lgbm_clf = lgb.sklearn.LGBMClassifier()
        lgbm_clf.fit(train_features, train_labels)
        # random forest
        # clf_5cv = svm.SVC(C=2, kernel='linear', gamma=20, decision_function_shape='ovr', probability=True)
        # clf_5cv.fit(train_features, train_labels)
        # svm
        # svm_clf = svm.SVC(C=2, kernel='linear', gamma=20, decision_function_shape='ovr', probability=True)
        # svm_clf.fit(train_features, train_labels)

        # lgbm evaluation
        y_pred_lgbm = lgbm_clf.predict(test_features)  # 测试集的预测标签

        tn, fp, fn, tp = confusion_matrix(test_labels, y_pred_lgbm).ravel()
        acc = accuracy_score(test_labels, y_pred_lgbm)  # acc
        ACC.append(acc)
        f1 = f1_score(test_labels, y_pred_lgbm)  # f1
        F1.append(f1)
        ppv = tp / (tp + fp)  # positive predictive value
        PPV.append(ppv)
        npv = tn / (tn + fn)  # negative predictive value
        NPV.append(npv)
        tpr = tp / (tp + fn)  # sensitivity / true positive rate
        TPR.append(tpr)
        tnr = tn / (tn + fp)  # specificity / true negative rate
        TNR.append(tnr)
        auc = roc_auc_score(test_labels, lgbm_clf.predict_proba(test_features)[:, 1])  # auc
        AUC.append(auc)
        mcc = matthews_corrcoef(test_labels, y_pred_lgbm)  # mcc Matthews correlation coefficient
        MCC.append(mcc)
        opm = (ppv + npv) * (tpr + tnr) * (acc + (1 + mcc) / 2) / 8  # opm
        OPM.append(opm)

        # save model
        #     joblib.dump(lgbm_clf, r'/home/Lucifer/synonymous/models/5-cv/lgbm_model{}.pkl'.format(i), compress=3)

        # print('svm:')
        # print("svm-测试集ACC：", acc)
        # print("svm-测试集F1：", f1)
        # print("svm-测试集PPV：", ppv)
        # print("svm-测试集NPV：", npv)
        # print("svm-测试集TPR：", tpr)
        # print("svm-测试集TNR：", tnr)
        # print("svm-测试集AUC：", auc)
        # print("svm-测试集MCC：", mcc)
        # print("svm-测试集OPM：", opm)

    ACC = np.array(ACC)
    # return np.mean(ACC)
    F1 = np.array(F1)
    NPV = np.array(NPV)
    TPR = np.array(TPR)
    TNR = np.array(TNR)
    AUC = np.array(AUC)
    MCC = np.array(MCC)
    OPM = np.array(OPM)
    PPV = np.array(PPV)
    TP = np.array(TP)
    TN = np.array(TN)
    FP = np.array(FP)
    FN = np.array(FN)
    print("CV上的平均值：")
    print("ACC:{}".format(np.mean(ACC)))
    print("F1:{}".format(np.mean(F1)))
    print("PPV:{}".format(np.mean(PPV)))
    print("NPV:{}".format(np.mean(NPV)))
    print("TPR:{}".format(np.mean(TPR)))
    print("TNR:{}".format(np.mean(TNR)))
    print("AUC:{}".format(np.mean(AUC)))
    print("MCC:{}".format(np.mean(MCC)))
    print("OPM:{}".format(np.mean(OPM)))


if __name__ == "__main__":
    max_test_acc = 0
    max_train_acc = 0
    feature_nums = [5, 10, 20, 30, 33, 50, 100, 200]
    features_importance = pd.read_csv("lgb_features_importance1001.csv")
    features_importance.sort_values(by="importance", ascending=False, inplace=True)
    features_importance.reset_index(inplace=True)
    features_importance = features_importance.loc[features_importance["importance"] != 0]
    all_features = features_importance["column"].tolist()

    features_selected = all_features[0: 33]
    blind_test(features_selected)





