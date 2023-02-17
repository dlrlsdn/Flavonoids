import pandas as pd
import glob

# pd.set_option('display.max_rows', None)

dpath = "/home/jjw/data"
path = '/code'
all_dpath = f"{dpath}/4066/*.csv"
import sys

sys.path.append(path)

from rdkit import Chem

# ################################################## 0_extract #########################################################
# Extract Smiles & Activity(Antioxidant) data from csv gained by PubChem
dfall = pd.DataFrame()
for i in glob.glob(f"{dpath}/4066/*.csv"):
    k = pd.read_csv(i)
    df = pd.DataFrame()
    df["Smiles"] = k.PUBCHEM_EXT_DATASOURCE_SMILES
    df["Activity"] = k.PUBCHEM_ACTIVITY_OUTCOME
    df = df.dropna(axis=0)
    dfall = pd.concat([dfall, df])

dfall.reset_index(inplace=True)
dfall.drop(["index"], axis=1, inplace=True)

# ############################################### 1_smi_to_inchikey ####################################################
# Transform Smiles to Inchikey which is unique value to Chemical compounds
ick = []
for x in dfall.Smiles:
    a = Chem.MolFromSmiles(x)
    b = Chem.MolToInchiKey(a)
    ick.append(b)

dfall["Inchikey"] = ick
ddf = dfall.drop_duplicates(['Inchikey'])
ddf.reset_index(inplace=True)
ddf.drop(["index"], axis=1, inplace=True)

# ################################################# 2_process ##########################################################
# Drop duplicated data with setting a standard for Inchikey
bool_list = dfall.duplicated(subset='Inchikey')
count = 0

tl = []
for y in bool_list:
    if y == True:
        tl.append(count)
    else:
        pass
    count = count + 1

il = []
for z in tl:
    il.append(dfall.Inchikey[z])

il = list(set(il))
fdf = pd.DataFrame()
for m in il:
    fdf = pd.concat([fdf, dfall[dfall['Inchikey'] == m]])

# write fdf to file(duplicated.csv)
fdf.to_csv(f"{dpath}/duplicated.csv", sep=',', index=False)

for t in il:
    idx = ddf[ddf['Inchikey'] == t].index
    ddf.drop(idx, axis='index', inplace=True)

ddf.reset_index(inplace=True)
ddf.drop(["index"], axis=1, inplace=True)

active = ddf[ddf['Activity'] == 'Active']
active.reset_index(inplace=True)
active.drop(["index"], axis=1, inplace=True)
active.to_csv(f"{dpath}/active.csv", sep=',', index=False)
active.to_csv(f"{dpath}/smi_active.csv", sep=',', index=False)
#
inactive = ddf[ddf['Activity'] == 'Inactive']
inactive.reset_index(inplace=True)
inactive.drop(["index"], axis=1, inplace=True)
inactive.to_csv(f"{dpath}/inactive.csv", sep=',', index=False)
inactive.to_csv(f"{dpath}/smi_inactive.csv", sep=',', index=False)
#
unspecified = ddf[ddf['Activity'] == 'Unspecified']
unspecified.reset_index(inplace=True)
unspecified.drop(["index"], axis=1, inplace=True)
unspecified.to_csv(f"{dpath}/unspecified.csv", sep=',', index=False)
unspecified.to_csv(f"{dpath}/smi_unspecified.csv", sep=',', index=False)
#
inconclusive = ddf[ddf['Activity'] == 'Inconclusive']
inconclusive.reset_index(inplace=True)
inconclusive.drop(["index"], axis=1, inplace=True)
inconclusive.to_csv(f"{dpath}/inconclusive.csv", sep=',', index=False)
inconclusive.to_csv(f"{dpath}/smi_inconclusive.csv", sep=',', index=False)

# ################################################# 3_save_csv #########################################################
# Write dropped(duplicated) datas to csv file
fdf.to_csv(f"{dpath}/duplicated_data", sep='\t', index=False)
