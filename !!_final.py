# Treat Natural Product Atlas data

# ############################################# Prepare & Process data #################################################
import pandas as pd

dpath = '/home/jjw/data'
cpath = '/home/jjw/PycharmProjects/pythonProject/code'
import sys

sys.path.append(f'{cpath}/process')

# Road raw data
d = pd.read_csv(f"{dpath}/prof/NPAtlas_download.tsv", sep='\t')
#
data = pd.DataFrame()
data["Compound"] = d.compound_names
data["Smiles"] = d.compound_smiles

# Smiles to RDkit
import jw_rdkit_smiles_to_fp as rstf

rdk = []
for k in data.Smiles.tolist():
    a = rstf.smi2fp(k)
    rdk.append(a)

data["RDkit"] = rdk

# Train data
pubchem = pd.read_csv(f"{dpath}/pubchem.csv")
flavonoid = pd.read_csv(f"{dpath}/flavonoids.csv")
flavonoid.drop(["RDkit"], axis=1, inplace=True)
train = pd.concat([pubchem, flavonoid])
train.reset_index(inplace=True)
train.drop(["index"], axis=1, inplace=True)
train_rdk = []
for k in train.Smiles.tolist():
    a = rstf.smi2fp(k)
    train_rdk.append(a)

train_y = train.Antioxidant.tolist()

# ################################################ Machine learning ####################################################
# Learning & Calculate probability
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

svc = SVC(probability=True)
lr = LogisticRegression()
xgb = XGBClassifier()
rf = RandomForestClassifier()

svc.fit(train_rdk, train_y)
prob_svc = svc.predict_proba(rdk)

lr.fit(train_rdk, train_y)
prob_lr = lr.predict_proba(rdk)

xgb.fit(train_rdk, train_y)
prob_xgb = xgb.predict_proba(rdk)

rf.fit(train_rdk, train_y)
prob_rf = rf.predict_proba(rdk)

# Organize data
import numpy as np

svcp = []
lrp = []
xgbp = []
rfp = []
for t in range(len(rdk)):
    svcp.append(prob_svc[t][1])
    lrp.append(prob_lr[t][1])
    xgbp.append(prob_xgb[t][1])
    rfp.append(prob_rf[t][1])

model_type = ['SVC', 'LR', 'XGB', 'RF']
ydf_col = data.Compound.tolist()
ydf = pd.DataFrame(np.array([svcp, lrp, xgbp, rfp]))
ydf.columns = ydf_col
ydf.index = model_type
ydf = ydf.T
ydf.reset_index(inplace=True)
ydf.rename(columns={'index': 'Compound'}, inplace=True)

# Choose
idx_svc = ydf.index[ydf.SVC >= 0.9].tolist()  # 358
idx_lr = ydf.index[ydf.LR >= 0.9].tolist()  # 1380
idx_xgb = ydf.index[ydf.XGB >= 0.9].tolist()  # 2228
idx_rf = ydf.index[ydf.RF >= 0.9].tolist()  # 48

# Check
a = list(set(idx_rf).intersection(idx_svc))  # 46
b = list(set(idx_rf).intersection(idx_lr))  # 44
c = list(set(idx_rf).intersection(idx_rf))  # 48
#
d = list(set(b).intersection(a))
e = list(set(b).intersection(c))

comp_list = []
for x in e:
    f = ydf.Compound[x]
    comp_list.append(f)

print(comp_list)

# ########################################### Make Network using Cytoscape #############################################
# variable e == comp_idx
comp_idx = [11395, 29701, 5637, 2181, 29703, 29704, 778, 15754, 26763, 5369, 6416, 14483, 3992, 11417, 5021, 19614,
            15908, 14894, 7215, 1840, 12340, 28981, 1979, 1471, 1984, 6850, 1349, 8137, 15947, 9678, 16719, 5842, 3539,
            4442, 32346, 32349, 25956, 18661, 20455, 1901, 1262, 17008, 3826, 2169]

# Make DataFrame
smi = []
name = []
for k in comp_idx:
    smi += data.Smiles[data.index == k].tolist()
    name += data.Compound[data.index == k].tolist()

ndf = pd.DataFrame()
ndf['Compound'] = name
ndf['Smiles'] = smi

# Made DataFrame for Cytoscape
import jw_rdkit_smiles_to_inchikey as rsti
import itertools
from rdkit import DataStructs

train = pd.read_csv(f'{dpath}/train.csv', sep=',')
train.drop(['Inchikey', 'RDkit', 'Antioxidant'], axis=1, inplace=True)
ndf2 = ndf.drop('Compound', axis=1)

ff = pd.concat([train, ndf2])
ff.reset_index(inplace=True)
ff.drop('index', axis=1, inplace=True)
ff.drop_duplicates(inplace=True)

tanimoto = pd.DataFrame()
tanimoto_ick1 = []
tanimoto_ick2 = []
tanimoto_value = []
for m in itertools.combinations(ff.Smiles.tolist(), 2):
    e = rstf.smi2fp(m[0])
    f = rstf.smi2fp(m[1])
    g = rsti.smi2ik(m[0])
    h = rsti.smi2ik(m[1])
    tanimoto_ick1.append(g)
    tanimoto_ick2.append(h)
    tanimoto_value.append(DataStructs.FingerprintSimilarity(e, f))

tanimoto["Inchikey_1"] = tanimoto_ick1
tanimoto["Inchikey_2"] = tanimoto_ick2
tanimoto["Tanimoto"] = tanimoto_value
# tanimoto.to_csv(f"{dpath}/cytoscape/train_npatlas.csv", sep=',', index=False)

# Make Search list file
ick = []
for n in ndf.Smiles.tolist():
    i = rsti.smi2ik(n)
    ick.append(i)

ndf['Inchikey'] = ick
sl = pd.DataFrame()
sl['Inchikey'] = ick
sl.to_csv(f'{dpath}/cytoscape/search_npatlas.csv', sep=',', index=False)

# ########################################## Make Heatmap using probabilities ##########################################
# Load data
dfs = pd.read_csv(f'{dpath}/prof/npatlas_prob.tsv', sep='\t', index_col=0)
dfs.reset_index(inplace=True)
dfs.rename(columns={'index': 'Compound'}, inplace=True)
dfs2 = dfs.loc[comp_idx, ['Compound', 'SVC', 'LR', 'XGB', 'RF']]

# Plot
import matplotlib.pyplot as plt
import numpy as np

model_type = ['SVC', 'LR', 'XGB', 'RF']
ydff_col = dfs2.Compound.tolist()
ydff = pd.DataFrame(np.array([dfs2.SVC.tolist(), dfs2.LR.tolist(), dfs2.XGB.tolist(), dfs2.RF.tolist()]))
ydff.columns = ydff_col
ydff.index = model_type
ydff = ydff.T  # Transpose
#
plt.figure(figsize=(7, 11))
plt.imshow(ydff.values.astype(float), cmap='OrRd')  # Essential to change values to float
plt.xticks(range(len(ydff.columns)), ydff.columns, fontsize=7)
plt.yticks(range(len(ydff.index)), ydff.index, fontsize=9)
plt.title("Heat_Map")
plt.colorbar()  # Make colorbar
plt.show()

# ##################################### Make Heatmap using Tanimoto similarity #########################################
# Plot HeatMap using Tanimoto (44 x 44 Matrix)
df = pd.read_csv(f'{dpath}/prof/npatlas_data_list.csv', sep=',')
df.sort_values('Inchikey', inplace=True)
ttanimoto_comp = df.Compound.tolist()
#
ttanimoto = pd.DataFrame()
ttanimoto_ick1 = []
ttanimoto_ick2 = []
ttanimoto_value = []
for m in itertools.product(df.Smiles.tolist(), repeat=2):
    e = rstf.smi2fp(m[0])
    f = rstf.smi2fp(m[1])
    g = rsti.smi2ik(m[0])
    h = rsti.smi2ik(m[1])
    ttanimoto_ick1.append(g)
    ttanimoto_ick2.append(h)
    ttanimoto_value.append(DataStructs.FingerprintSimilarity(e, f))

ttanimoto["Inchikey_1"] = ttanimoto_ick1
ttanimoto["Inchikey_2"] = ttanimoto_ick2
ttanimoto["Tanimoto"] = ttanimoto_value
ttanimoto.sort_values(['Inchikey_1', 'Inchikey_2'], inplace=True)
# Make proper Matrix similar to diagonal matrix for HeatMap
mtx_hm = pd.DataFrame()
for n in range(0, 1936, 44):
    a = ttanimoto.iloc[n:n + 44, 2].tolist()
    mtx_hm[f'{n}'] = a

mtx_hm.index = ttanimoto_comp
mtx_hm.columns = ttanimoto_comp
mtx_hm.to_csv(f"{dpath}/prof/matrx_npatlas.csv", sep=',', index=False)

# Plot
import scipy.cluster.hierarchy as sch

fig = plt.figure(figsize=(90, 90))

# Dendrogram which comes to the right
ax = fig.add_axes([0.351, 0.4, 0.1, 0.4])
dend_l = sch.linkage(mtx_hm.values, method='complete')
dend = sch.dendrogram(dend_l, orientation='left', labels=ttanimoto_comp)
ax.set_xticks([])
ax.set_yticks([])
# Extract arranged compound list
dl = dend['ivl']
dl.reverse()
# Arrange Heatmap
mtx_hm = mtx_hm.reindex(dl)
mtx_hm = mtx_hm.reindex(dl, axis=1)

# Heatmap
ax_hm = fig.add_axes([0.3, 0.4, 0.5, 0.5])
ax_hm.imshow(mtx_hm.values, cmap='OrRd', interpolation='none')  # interpolation == 보간법
# Set xticks
ax_hm.set_xticks(range(len(dl)))
ax_hm.set_xticklabels(dl, rotation=90, fontsize=7)
ax_hm.xaxis.tick_bottom()
# Set yticks
ax_hm.set_yticks(range(len(dl)))
ax_hm.set_yticklabels(dl, fontsize=7)
ax_hm.yaxis.tick_right()
# Set colorbar
k = ax_hm.imshow(mtx_hm.values, cmap='OrRd', interpolation='none')
cbar = plt.colorbar(k, location='top', shrink=0.33)  # Make colorbar
cbar.ax.tick_params(labelsize=8)
# Set title, ticks
plt.title("Heat_Map", pad=50, fontsize=8)
plt.tick_params(axis='x', length=3, pad=2)
plt.tick_params(axis='y', length=3, pad=2)
#
plt.show()
