# Prepare tuning

cpath = '/home/jjw/PycharmProjects/pythonProject/code'
import sys

# sys.path.append(f'{cpath}/process')
sys.path.append(f'{cpath}/uf')
import mknumlist as mnl

# #### Make param_grid for SVM
C_svm = []  # default == 1
degree_svm = []  # default == 3
gamma_svm = ['scale', 'auto']  # default == 'scale'; float 가능

# Select proper range of numbers
C_svm = mnl.mki(1, 5, C_svm, 10)
degree_svm = mnl.mki(1, 1, degree_svm, 20)

# hyperparameter dict
hp_svc = {'C': C_svm, 'gamma': gamma_svm, 'kernel': ['linear', 'rbf', 'sigmoid']}

# #### Make param_grid for Logistic Regression
# default --> [class_weight, dual, warm_start]
C_lr = []  # default == 1
tol = []  # default == 0.0001
max_iter = []  # default == 100

# Select proper range of numbers
C_lr = mnl.mki(1, 5, C_lr, 10)
tol = mnl.mkf(0.0001, 0.0001, tol, 10)
# max_iter = mnl.mki(90, 10, max_iter, 15)
max_iter = [100]

# hyperparameter dict
hp_lr = {'C': C_lr,
         'penalty': ['l1', 'l2', 'elasticnet', 'None'],
         'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'],
         'tol': tol,
         'max_iter': max_iter}

# #### Make param_grid for XGBoost
# esr = []        # default == None
# default --> [scale_pos_weight]
# eval_set --> Parameters: { "eval_set" } might not be used
gamma_xgb = []  # default == 0
mcw = []  # default == 1
eta = []  # default == 0.3
alpha = []  # default == 0
lamda = []  # default == 1
sub_sample = []  # default == 1
colsample_bytree = []  # default == 1
colsample_bylevel = []  # default == 1
colsample_bynode = []  # default == 1
n_est_xgb = []  # default == 10
max_depth = []  # default == 6

# Select proper range of numbers
gamma_xgb = mnl.mki(0, 1, gamma_xgb, 10)
mcw = mnl.mki(1, 1, mcw, 10)
eta = mnl.mki(0.1, 0.1, eta, 10)
alpha = mnl.mki(0, 0.1, alpha, 10)
lamda = mnl.mki(1, 1, lamda, 10)
sub_sample = mnl.mki(1, 1, sub_sample, 10)
colsample_bytree = mnl.mki(1, 1, colsample_bytree, 10)
colsample_bylevel = mnl.mki(1, 1, colsample_bylevel, 10)
colsample_bynode = mnl.mki(1, 1, colsample_bynode, 10)
n_est_xgb = mnl.mki(8, 1, n_est_xgb, 12)
max_depth = mnl.mki(4, 1, max_depth, 10)

# hyperparameter dict
hp_xgb = {'gamma': gamma_xgb, 'booster': ['gbtree', 'gblinear'], 'eta': eta,
          'n_estimators': n_est_xgb,
          'min_child_weight': mcw,
          'reg_lambda': lamda, 'reg_alpha': alpha,
          'max_depth': max_depth,
          'subsample': sub_sample,
          'colsample_bytree': colsample_bytree,
          'colsample_bylevel': colsample_bylevel,
          'colsample_bynode': colsample_bynode,
          'early_stopping_rounds': [None],
          'objective': ['binary:logistic'],
          'eval_metric': ['error'],
          'base_score': [0.5],
          'verbosity': [2]}

# #### Make param_grid for Random Forest
# max_depth = []      # default = None
# max_samples = []        # default == None
# default --> [class_weight, warm_start, ccp_alpha, max_depth, max_leaf_nodes, max_samples]
n_est_rf = []  # default == 100
mss = []  # default == 2
msl = []  # default == 1
mid = []  # default == 0

# Select proper range of numbers
n_est_rf = mnl.mki(90, 10, n_est_rf, 20)
mss = mnl.mki(1, 1, mss, 10)
msl = mnl.mki(1, 1, msl, 10)
mid = mnl.mki(0, 1, mid, 10)

# hyperparameter dict
hp_rf = {'n_estimators': n_est_rf,
         'min_samples_split': mss,
         'min_samples_leaf': msl,
         'max_features': ['sqrt', 'log2'],
         'criterion': ['gini', 'entropy', 'log_loss'],
         'min_weight_fraction_leaf': [0.0],
         'min_impurity_decrease': mid,
         'bootstrap': ['True'],
         'oob_score': ['True', 'False']}

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd
import pickle

# Load Train data
dpath = '/home/jjw/data'
import jw_rdkit_smiles_to_fp as rstf

train = pd.read_csv(f'{dpath}/train.csv')
train.reset_index(inplace=True)
train.drop(["index", 'RDkit', 'Inchikey'], axis=1, inplace=True)
train_rdk = []
for m in train.Smiles.tolist():
    a = rstf.smi2fp(m)
    train_rdk.append(a)

train_y = train.Antioxidant.tolist()

# Declare Model
svc = SVC()
lr = LogisticRegression()
xgb = XGBClassifier()
rf = RandomForestClassifier()

# Tuning for each Model
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True)

# ##################################################### SVM ############################################################
# Accuruacy
a_svc = GridSearchCV(estimator=svc, param_grid=hp_svc, cv=kf, scoring='accuracy')
a_svc.fit(train_rdk, train_y)

# Sensitivity
se_svc = GridSearchCV(estimator=svc, param_grid=hp_svc, cv=kf, scoring='recall')
se_svc.fit(train_rdk, train_y)

# Preicision
pr_svc = GridSearchCV(estimator=svc, param_grid=hp_svc, cv=kf, scoring='precision')
pr_svc.fit(train_rdk, train_y)

# ROC_AUC
auc_svc = GridSearchCV(estimator=svc, param_grid=hp_svc, cv=kf, scoring='roc_auc')
auc_svc.fit(train_rdk, train_y)

# F1 score
f1_svc = GridSearchCV(estimator=svc, param_grid=hp_svc, cv=kf, scoring='f1')
f1_svc.fit(train_rdk, train_y)

# ###################################################### LR ############################################################
# Accuruacy
a_lr = GridSearchCV(estimator=lr, param_grid=hp_lr, cv=kf, scoring='accuracy')
a_lr.fit(train_rdk, train_y)

# Sensitivity
se_lr = GridSearchCV(estimator=lr, param_grid=hp_lr, cv=kf, scoring='recall')
se_lr.fit(train_rdk, train_y)

# Preicision
pr_lr = GridSearchCV(estimator=lr, param_grid=hp_lr, cv=kf, scoring='precision')
pr_lr.fit(train_rdk, train_y)

# ROC_AUC
auc_lr = GridSearchCV(estimator=lr, param_grid=hp_lr, cv=kf, scoring='roc_auc')
auc_lr.fit(train_rdk, train_y)

# F1 score
f1_lr = GridSearchCV(estimator=lr, param_grid=hp_lr, cv=kf, scoring='f1')
f1_lr.fit(train_rdk, train_y)

# ##################################################### XGB ############################################################
# Accuruacy
a_xgb = GridSearchCV(estimator=xgb, param_grid=hp_xgb, cv=kf, scoring='accuracy')
a_xgb.fit(train_rdk, train_y)

# Sensitivity
se_xgb = GridSearchCV(estimator=xgb, param_grid=hp_xgb, cv=kf, scoring='recall')
se_xgb.fit(train_rdk, train_y)

# Preicision
pr_xgb = GridSearchCV(estimator=xgb, param_grid=hp_xgb, cv=kf, scoring='precision')
pr_xgb.fit(train_rdk, train_y)

# ROC_AUC
auc_xgb = GridSearchCV(estimator=xgb, param_grid=hp_xgb, cv=kf, scoring='roc_auc')
auc_xgb.fit(train_rdk, train_y)

# F1 score
f1_xgb = GridSearchCV(estimator=xgb, param_grid=hp_xgb, cv=kf, scoring='f1')
f1_xgb.fit(train_rdk, train_y)

# ###################################################### RF ############################################################
# Accuruacy
a_rf = GridSearchCV(estimator=rf, param_grid=hp_rf, cv=kf, scoring='accuracy')
a_rf.fit(train_rdk, train_y)

# Sensitivity
se_rf = GridSearchCV(estimator=rf, param_grid=hp_rf, cv=kf, scoring='recall')
se_rf.fit(train_rdk, train_y)

# Preicision
pr_rf = GridSearchCV(estimator=rf, param_grid=hp_rf, cv=kf, scoring='precision')
pr_rf.fit(train_rdk, train_y)

# ROC_AUC
auc_rf = GridSearchCV(estimator=rf, param_grid=hp_rf, cv=kf, scoring='roc_auc')
auc_rf.fit(train_rdk, train_y)

# F1 score
f1_rf = GridSearchCV(estimator=rf, param_grid=hp_rf, cv=kf, scoring='f1')
f1_rf.fit(train_rdk, train_y)

# best hyperparameter 획득
# print(~.best_params_)
# print(~.best_score_)
