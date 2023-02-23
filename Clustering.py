# Process & Prepare data for Cytoscape

from rdkit import DataStructs
from rdkit import Chem
import itertools
import pandas as pd
import sys

path = '/home/jjw/PycharmProjects/pythonProject/code'
dpath = '/home/jjw/data'
sys.path.append(f"{path}/process")
sys.path.append(f"{dpath}")

import jw_rdkit_smiles_to_fp as rstf

# ################################# Calculate similarity using Tanimoto calculation ####################################
# Cleaning up data
active = pd.read_csv(f"{dpath}/active.csv")
active.drop(['Inchikey'], axis=1, inplace=True)
act_list = []
# type_act = []
for m in range(448):
    act_list.append(1)
    # type_act.append("Training_Active")

active["Antioxidant"] = act_list
# active["Type"] = type_act
active.drop(['Activity'], axis=1, inplace=True)
#
inactive = pd.read_csv(f"{dpath}/inactive.csv")
inactive.drop(['Inchikey'], axis=1, inplace=True)
inact_list = []
# type_inact = []
for n in range(466):
    inact_list.append(0)
    # type_inact.append("Training_Inactive")

inactive["Antioxidant"] = inact_list
# inactive["Type"] = type_inact
inactive.drop(['Activity'], axis=1, inplace=True)
#
flavonoids = pd.read_csv(f"{dpath}/flavonoids.csv")
flavonoids.drop(['RDkit'], axis=1, inplace=True)
# type_flavo = []
# for j in range(32):
#     type_flavo.append("Test_Flavonoids")

# flavonoids["Type"] = type_flavo
#
fdf = pd.concat([active, inactive])
fdf = pd.concat([fdf, flavonoids])
fdf.reset_index(inplace=True)
fdf.drop(["index"], axis=1, inplace=True)

# Prepare RDkit data
rdk = []
for l in fdf.Smiles.tolist():
    x = rstf.smi2fp(l)
    rdk.append(x)

# Make Tanimoto dataframe
tanimoto_df = pd.DataFrame()
fl_1 = []
fl_2 = []
tanimoto = []
for k in itertools.combinations(rdk, 2):
    a = DataStructs.FingerprintSimilarity(k[0], k[1])
    fl_1.append(k[0])
    fl_2.append(k[1])
    tanimoto.append(a)

tanimoto_df["Flavonoids_1"] = fl_1
tanimoto_df["Flavonoids_2"] = fl_2
tanimoto_df["Tanimoto"] = tanimoto

# Save Tanimoto dataframe
tanimoto_df.to_csv(f"{dpath}/tanimoto_df.csv", sep=',', index=False)

# Set standard value
value = tanimoto_df.Tanimoto.tolist()
value = list(set(value))
value.sort()

# ############################################ Smiles pair for Cytoscape ###############################################
smi_pair = pd.DataFrame()
smi_1 = []
smi_2 = []
for t in itertools.combinations(fdf.Smiles.tolist(), 2):
    smi_1.append(t[0])
    smi_2.append(t[1])

smi_pair["Smiles_1"] = smi_1
smi_pair["Smiles_2"] = smi_2

smi_pair.to_csv(f"{dpath}/smi_pair.csv", sep=',', index=False)

# Smiles pair + Tanimoto
smi_tanimoto = pd.DataFrame()
st_1 = []
st_2 = []
t_val = []
for x in itertools.combinations(fdf.Smiles.tolist(), 2):
    st_1.append(x[0])
    st_2.append(x[1])
    a = rstf.smi2fp(x[0])
    b = rstf.smi2fp(x[1])
    t_val.append(DataStructs.FingerprintSimilarity(a, b))

smi_tanimoto["Smiles_1"] = st_1
smi_tanimoto["Smiles_2"] = st_2
smi_tanimoto["Tanimoto"] = t_val

smi_tanimoto.to_csv(f"{dpath}/smi_tanimoto.csv", sep=',', index=False)

# ########################################### Inchikey pair for Cytoscape ##############################################

import jw_rdkit_smiles_to_inchikey as rsti

std = pd.read_csv(f"{dpath}/smi_tanimoto.csv")

ik1 = []
ik2 = []
for y in std.itertuples():
    ik1.append(rsti.smi2ik(y[1]))
    ik2.append(rsti.smi2ik(y[2]))

std.drop(["Smiles_1"], axis=1, inplace=True)
std.drop(["Smiles_2"], axis=1, inplace=True)
std["Inchikey_1"] = ik1
std["Inchikey_2"] = ik2
# Arrange columns
std = std[["Inchikey_1", "Inchikey_2", "Tanimoto"]]

std.to_csv(f"{dpath}/ik_tanimoto.csv", sep=',', index=False)

# ######################################## Extract Inchikey of flavonoids ##############################################

fik = []
for z in flavonoids.Smiles.tolist():
    a = rsti.smi2ik(z)
    fik.append(a)

flavonoids["Inchikey"] = fik
