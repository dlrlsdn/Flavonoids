#!/usr/bin/env python
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

def smi2fp(smiles, radius = 2):
	try:
		mol = Chem.MolFromSmiles(smiles)
		fp_bv = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius)
		# nBits = 2048 (default), radius = 2 (ECFP4-like fingerprint)
	except Exception as e:
		print(e)
		fp_bv = ''
	return fp_bv


fp = smi2fp("C1NCN1.C1NCN1")

