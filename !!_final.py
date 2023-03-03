# Load & Process data / Learning / Find & Plot antioxidant compounds
dpath = '/home/jjw/data'
cpath = '/home/jjw/PycharmProjects/pythonProject/code'
# pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", None)
import sys

sys.path.append(f"{cpath}/process")
import pandas as pd
import jw_rdkit_smiles_to_fp as rstf

# ################################################## Load & Process data ###############################################
train = pd.read_csv(f"{dpath}/train.csv")
train.drop(["RDkit"], axis=1, inplace=True)
train_rdk = []
for k in train.Smiles.tolist():
    a = rstf.smi2fp(k)
    train_rdk.append(a)

test = pd.read_csv(f"{dpath}/prof/core_mhy.csv")
test.drop(["RDkit"], axis=1, inplace=True)
test_rdk = []
for m in test.Smiles.tolist():
    b = rstf.smi2fp(m)
    test_rdk.append(b)

train_y = train.Antioxidant.tolist()

# ######################################### Learning & Finding antioxidant compounds ###################################
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

svc = SVC()
lr = LogisticRegression()
xgb = XGBClassifier()
rf = RandomForestClassifier()

svc.fit(train_rdk, train_y)
y_svc = svc.predict(test_rdk)

lr.fit(train_rdk, train_y)
y_lr = lr.predict(test_rdk)

xgb.fit(train_rdk, train_y)
y_xgb = xgb.predict(test_rdk)

rf.fit(train_rdk, train_y)
y_rf = rf.predict(test_rdk)

# Arrange data for easily seeing
import numpy as np

model_type = ['SVC', 'LR', 'XGB', 'RF']
ydf_col = test.Compound.tolist()
ydf = pd.DataFrame(np.array([y_svc, y_lr, y_xgb, y_rf]))
ydf.columns = ydf_col
ydf.insert(0, 'Compound', model_type)

# Save
ydf.to_csv(f"{dpath}/prof/compound_mhy.csv", sep=',', index=False)

# ############################ Select compounds predicted to be antioxidant from 4 ML models ###########################
data = pd.read_csv(f"{dpath}/prof/compound_mhy.csv")
comp_list = []
for k in range(len(data.values[0])):
    try:
        v = []
        for m in range(4):
            v.append(data.values[m][k])
        if sum(v) >= 3.5:
            comp_list.append(data.columns[k])
        else:
            pass
    except:
        pass

for x in comp_list:
    print(x)

result1 = ["hypolaetin 7-O-xylopyranosdie", "quercetin-7-O-β-D-rhamnoside", "yuanhuanin", "triumbelletin"]
index_of_result1 = [1, 11, 24, 29]

# ################################### Calculate probabilities of all test compounds ####################################
svc = SVC(probability=True)  # SVC demands you "probability=True" to calculate probability
lr = LogisticRegression()
xgb = XGBClassifier()
rf = RandomForestClassifier()

svc.fit(train_rdk, train_y)
prob_svc = svc.predict_proba(test_rdk)

lr.fit(train_rdk, train_y)
prob_lr = lr.predict_proba(test_rdk)

xgb.fit(train_rdk, train_y)
prob_xgb = xgb.predict_proba(test_rdk)

rf.fit(train_rdk, train_y)
prob_rf = rf.predict_proba(test_rdk)

# ################################################### Plot Heatmap #####################################################
import matplotlib.pyplot as plt
import numpy as np

svcp = []
lrp = []
xgbp = []
rfp = []
for t in range(41):
    svcp.append(prob_svc[t][1])
    lrp.append(prob_lr[t][1])
    xgbp.append(prob_xgb[t][1])
    rfp.append(prob_rf[t][1])

ydf_col = test.Compound.tolist()
ydf = pd.DataFrame(np.array([svcp, lrp, xgbp, rfp]))
ydf.columns = ydf_col
ydf.insert(0, 'Compound', model_type)
ydf = ydf.T  # Transpose
ydf.drop(["Compound"], axis=0, inplace=True)
ydf.columns = model_type

plt.figure(figsize=(5, 11))
plt.imshow(ydf.values.astype(float), cmap='OrRd')  # Essential to change values to float
plt.xticks(range(len(ydf.columns)), ydf.columns, fontsize=7)
plt.yticks(range(len(ydf.index)), ydf.index, fontsize=9)
plt.title("Heat_Map")
plt.colorbar()  # Make colorbar
index_of_result1 = [1, 11, 24, 29]
for e in index_of_result1:
    plt.gca().get_yticklabels()[e].set_color('blue')  # 특정 y값만 색상 변경

plt.show()

# ############################################ Prepare data for Cytoscape ##############################################
import jw_rdkit_smiles_to_inchikey as rsti

# Make check list of Antioxidant compounds in train set
act_ick = []
for k in range(len(train)):
    if train.Antioxidant[k] == 1:
        act_ick.append(train.Inchikey[k])

df_act = pd.DataFrame(act_ick, columns=['Inchikey'])
df_act.to_csv(f"{dpath}/cytoscape/check_act.csv", sep=',', index=False)

# Make check list of Antioxidant compounds in test set
ick_list = []
smi_list = []
for m in index_of_result1:
    ick_list.append(rsti.smi2ik(test.Smiles[m]))
    smi_list.append(test.Smiles[m])

df_test = pd.DataFrame({'Inchikey': ick_list, 'Smiles': smi_list})
test_check = pd.DataFrame(df_test.Inchikey.tolist(), columns=['Inchikey'])
test_check.to_csv(f"{dpath}/cytoscape/check_test.csv", sep=',', index=False)  # For search

# Make DataFrame consist of train and selected test
df_train = pd.DataFrame({'Inchikey': train.Inchikey.tolist(), 'Smiles': train.Smiles.tolist()})
dff = pd.concat([df_test, df_train])

# Calculate Tanimoto values
import itertools
from rdkit import DataStructs

c1 = []
c2 = []
tnm = []
for n in itertools.combinations(dff.Smiles.tolist(), 2):
    b = rstf.smi2fp(n[0])
    c = rstf.smi2fp(n[1])
    d = rsti.smi2ik(n[0])
    e = rsti.smi2ik(n[1])
    tnm.append(DataStructs.FingerprintSimilarity(b, c))
    c1.append(d)
    c2.append(e)

tanimoto_ddf = pd.DataFrame({'Compound1': c1, 'Compound2': c2, 'Tanimoto': tnm})
tanimoto_ddf.to_csv(f"{dpath}/cytoscape/train_selected.csv", sep=',', index=False)

# Check Tanimoto values between selected compounds
tnm2 = []
cc1 = []
cc2 = []
for x in itertools.combinations(df_test.Smiles.tolist(), 2):
    a = rstf.smi2fp(x[0])
    b = rstf.smi2fp(x[1])
    c = rsti.smi2ik(x[0])
    d = rsti.smi2ik(x[1])
    tnm2.append(DataStructs.FingerprintSimilarity(a, b))
    cc1.append(c)
    cc2.append(d)

tanimoto_ddf2 = pd.DataFrame({'Compound1': cc1, 'Compound2': cc2, 'Tanimoto': tnm2})
tanimoto_ddf2.to_csv(f"{dpath}/cytoscape/selected_tanimoto.csv", sep=',', index=False)
