#!/usr/bin/env python
from rdkit import Chem


def smi2ik(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        inchik = Chem.MolToInchiKey(mol)
    except Exception as e:
        print(e)
        inchik = ''
    return inchik


ik = smi2ik("C1NCN1.C1NCN1")
