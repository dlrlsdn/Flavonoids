# Machine Learning
dpath = "/home/jjw/data"
cpath = "/home/jjw/PycharmProjects/pythonProject/code"

# Add system path
import sys

sys.path.append(f"{cpath}/sklearn")
sys.path.append(f"{cpath}/process")
sys.path.append(f"{dpath}")

import evaluate_function as ef
import jw_rdkit_smiles_to_fp as rstf
import pandas as pd
import pickle

# ############################################### 0_smi_to_rdkit #######################################################
# Prepare data
active = pd.read_csv(f"{dpath}/smi_active.csv")
inactive = pd.read_csv(f"{dpath}/smi_inactive.csv")

rdk_active = []
for k in active.Smiles:
    a = rstf.smi2fp(k)
    rdk_active.append(a)

active['RDkit'] = rdk_active
active.drop(['Smiles', 'Inchikey'], axis=1, inplace=True)

rdk_inactive = []
for m in inactive.Smiles:
    b = rstf.smi2fp(m)
    rdk_inactive.append(b)

inactive['RDkit'] = rdk_inactive
inactive.drop(['Smiles', 'Inchikey'], axis=1, inplace=True)
df = pd.concat([active, inactive])
bool = []
for n in df.Activity:
    if n == "Active":
        bool.append(1)
    else:
        bool.append(0)

df["Antioxidant"] = bool
df.drop(["Activity"], axis=1, inplace=True)
df.reset_index(inplace=True)
df.drop(["index"], axis=1, inplace=True)

# ################################################### 1_learning #######################################################
# Import tools for Machine Learning
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split as tts

# Load binary data
out2 = open(f"{dpath}/data.dat", 'rb')
data = pickle.load(out2)
bdf = pd.DataFrame(data, columns=['RDkit', 'Antioxidant'])

# Declare Models
svc = SVC()
lr = LogisticRegression()
xgb = XGBClassifier()
rf = RandomForestClassifier()

# Dataframe Labels
colmn_svc = ['SVC_Q', 'SVC_SE', 'SVC_SP', 'SVC_PRE', 'SVC_ROAUC', 'SVC_F1']
colmn_lr = ['LR_Q', 'LR_SE', 'LR_SP', 'LR_PRE', 'LR_ROAUC', 'LR_F1']
colmn_xgb = ['XGB_Q', 'XGB_SE', 'XGB_SP', 'XGB_PRE', 'XGB_ROAUC', 'XGB_F1']
colmn_rf = ['RF_Q', 'RF_SE', 'RF_SP', 'RF_PRE', 'RF_ROAUC', 'RF_F1']

# Number of used Models
modeln = ['SVC', 'LogisticRegression', 'XGBoost', 'RandomForest']

# Number of used Values
scoren = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'ROAUC', 'F1 socre']

# Learning
outall = []
for z in range(1000):
    print(z)
    # ttsplit
    xs_train, xs_test, ys_train, ys_test = tts(bdf.RDkit, bdf.Antioxidant, test_size=0.2, shuffle=True,
                                               stratify=bdf.Antioxidant)
    xs_train = xs_train.tolist()
    xs_test = xs_test.tolist()
    ys_train = ys_train.tolist()
    ys_test = ys_test.tolist()
    # SVM
    svc.fit(xs_train, ys_train)
    svc_pred = svc.predict(xs_test)
    # LogisticRegression
    lr.fit(xs_train, ys_train)
    lr_pred = lr.predict(xs_test)
    # XGBoost
    xgb.fit(xs_train, ys_train)
    xgb_pred = xgb.predict(xs_test)
    # RandomForest
    rf.fit(xs_train, ys_train)
    rf_pred = rf.predict(xs_test)
    # data_evaluation
    dsvc = [qsvc, sesvc, spsvc, precisionsvc, roaucsvc, fsvc] = [ef.gq(ys_test, svc_pred), ef.gSE(ys_test, svc_pred),
                                                                 ef.gSP(ys_test, svc_pred), ef.gpr(ys_test, svc_pred),
                                                                 ef.roauc(ys_test, svc_pred), ef.gfs(ys_test, svc_pred)]
    dlr = [qlr, selr, splr, precisionlr, roauclr, flr] = [ef.gq(ys_test, lr_pred), ef.gSE(ys_test, lr_pred),
                                                          ef.gSP(ys_test, lr_pred), ef.gpr(ys_test, lr_pred),
                                                          ef.roauc(ys_test, lr_pred), ef.gfs(ys_test, lr_pred)]
    dxgb = [qxgb, sexgb, spxgb, precisionxgb, roaucxgb, fxgb] = [ef.gq(ys_test, xgb_pred), ef.gSE(ys_test, xgb_pred),
                                                                 ef.gSP(ys_test, xgb_pred), ef.gpr(ys_test, xgb_pred),
                                                                 ef.roauc(ys_test, xgb_pred), ef.gfs(ys_test, xgb_pred)]
    dfr = [qrf, serf, sprf, precisionrf, roaucrf, frf] = [ef.gq(ys_test, rf_pred), ef.gSE(ys_test, rf_pred),
                                                          ef.gSP(ys_test, rf_pred), ef.gpr(ys_test, rf_pred),
                                                          ef.roauc(ys_test, rf_pred), ef.gfs(ys_test, rf_pred)]
    datalist = [dsvc, dlr, dxgb, dfr]
    # dataframe
    outs = []
    for k in range(len(modeln)):
        for a in range(len(scoren)):
            outs.append([modeln[k]] + [scoren[a]] + [datalist[k][a]])
    outall = outall + outs

out2.close()
dfall = pd.DataFrame(outall, columns=['Model', 'Score', 'Value'])

# ################################################### 2_plot ###########################################################
# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

sns.boxplot(x='Score', y='Value', hue='Model', data=dfall)
plt.title('Predict Antioxidant')
plt.ylim(0.75, 1)
plt.show()

# ############################################### 3_pre_real_test ######################################################
# Import cirpy which transforms Cas number to SMILES
import cirpy

# Prepare Cas-number that I want to transform to SMILES
oh1 = ['6068-76-4', '55977-09-8', '14919-49-4', '6665-69-6', '108238-41-1', '492-00-2']
oh1_ao = [1, 1, 1, 0, 1, 1]
oh2 = ['6068-78-6', '548-83-4', '253195-19-6', '151698-64-5', '2034-65-3']
oh2_ao = [1, 1, 1, 1, 1]
oh3 = ['92439-38-8', '142646-44-4', '108239-98-1', '263365-54-4', '260063-28-3', '528-48-3', '480-15-9', '520-18-3',
       '1429-28-3']
oh3_ao = [0, 0, 0, 1, 1, 1, 1, 1, 0]
oh4 = ['490-31-3', '4324-55-4', '527-95-7', '480-16-0', '92519-95-4', '117-39-5', '28449-61-8', '80710-48-1',
       '74514-47-9', '489-58-7']
oh4_ao = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
oh5 = ['529-44-2', '90-18-6', '489-35-0', '4431-48-5']
oh5_ao = [1, 1, 1, 0]
oh6 = ['577-24-2', '87926-83-8']
oh6_ao = [0, 1]

cas = oh1 + oh2 + oh3 + oh4 + oh5 + oh6
cas_ao = oh1_ao + oh2_ao + oh3_ao + oh4_ao + oh5_ao + oh6_ao

smiles = []
for i in cas:
    a = cirpy.resolve(i, 'smiles')
    smiles.append(a)

index = []
for idx, smi in enumerate(smiles):
    if smi == None:
        index.append(idx)
    else:
        pass

error = []
for k in index:
    error.append(cas[k])
    print(cas[k])

# SMILES of Error Cas number
smi_dic = {'6068-76-4': 'C1=CC=C(C(=C1)C2=C(C(=O)C3=CC=CC=C3O2)O)O',
           '6665-69-6': 'C1=CC=C(C=C1)C2=C(C(=O)C3=C(C=CC=C3O2)O)O',
           '108238-41-1': 'C1=CC=C(C=C1)C2=C(C(=O)C3=C(O2)C=CC(=C3)O)O',
           '1429-28-3': 'C1=CC(=CC=C1C2=C(C(=O)C3=C(O2)C(=C(C=C3)O)O)O)O',
           '28449-61-8': 'C1=C(C=C(C=C1O)O)C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O',
           '577-24-2': 'C1=C(C=C(C(=C1O)O)O)C2=C(C(=O)C3=C(O2)C(=C(C=C3O)O)O)O',
           '87926-83-8': 'C1=CC(=C(C=C1C2=C(C(=O)C3=C(C(=C(C(=C3O2)O)O)O)O)O)O)O'}
smi_dic_oh = [1, 0, 1, 0, 0, 0, 1]

smiles.append(smi_dic['6068-76-4'])
smiles.append(smi_dic['6665-69-6'])
smiles.append(smi_dic['108238-41-1'])
smiles.append(smi_dic['1429-28-3'])
smiles.append(smi_dic['28449-61-8'])
smiles.append(smi_dic['577-24-2'])
smiles.append(smi_dic['87926-83-8'])
antioxidant = cas_ao + smi_dic_oh

df = pd.DataFrame({'Smiles': smiles, 'Antioxidant': antioxidant})
df = df.dropna()
df.reset_index(inplace=True)
df.drop(["index"], axis=1, inplace=True)

rdk = []
for k in df.Smiles:
    a = rstf.smi2fp(k)
    rdk.append(a)

df['RDkit'] = rdk

# ################################################# 4_real_test ########################################################
# Load binary data
out2 = open(f"{dpath}/data.dat", 'rb')
data = pickle.load(out2)
bdf = pd.DataFrame(data, columns=['RDkit', 'Antioxidant'])

x = df.RDkit.tolist()
y = df.Antioxidant.tolist()

# Learning & Prediction
svc = SVC()
lr = LogisticRegression()
xgb = XGBClassifier()
rf = RandomForestClassifier()

svc_x = bdf.RDkit.tolist()
svc_y = bdf.Antioxidant.tolist()
svc.fit(svc_x, svc_y)
y_svc = svc.predict(x)

lr_x = bdf.RDkit.tolist()
lr_y = bdf.Antioxidant.tolist()
lr.fit(lr_x, lr_y)
y_lr = lr.predict(x)

xgb_x = bdf.RDkit.tolist()
xgb_y = bdf.Antioxidant.tolist()
xgb.fit(xgb_x, xgb_y)
y_xgb = xgb.predict(x)

rf_x = bdf.RDkit.tolist()
rf_y = bdf.Antioxidant.tolist()
rf.fit(rf_x, rf_y)
y_rf = rf.predict(x)

out2.close()

# Arrange data for easily seeing
import numpy as np

set_svc = [ef.gq(y, y_svc), ef.gSE(y, y_svc), ef.gSP(y, y_svc), ef.gpr(y, y_svc), ef.roauc(y, y_svc), ef.gfs(y, y_svc)]
set_lr = [ef.gq(y, y_lr), ef.gSE(y, y_lr), ef.gSP(y, y_lr), ef.gpr(y, y_lr), ef.roauc(y, y_lr), ef.gfs(y, y_lr)]
set_xgb = [ef.gq(y, y_xgb), ef.gSE(y, y_xgb), ef.gSP(y, y_xgb), ef.gpr(y, y_xgb), ef.roauc(y, y_xgb), ef.gfs(y, y_xgb)]
set_rf = [ef.gq(y, y_rf), ef.gSE(y, y_rf), ef.gSP(y, y_rf), ef.gpr(y, y_rf), ef.roauc(y, y_rf), ef.gfs(y, y_rf)]

column_name = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'ROAUC', 'F1']
model_types = ['SVC', 'LR', 'XGB', 'RF']
test = pd.DataFrame(np.array([set_svc, set_lr, set_xgb, set_rf]))
test.columns = column_name
test.insert(0, 'Model', model_types)

ydf = pd.DataFrame(np.array([y, y_svc, y_lr, y_xgb, y_rf]))
ydf.insert(0, 'Model', ['Real'] + model_types)
