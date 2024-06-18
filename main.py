import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from pathlib import Path

from chemprop import data, featurizers, models

checkpoint_path = r'encoder/example_model_v2_regression_mol.ckpt'
mpnn = models.MPNN.load_from_checkpoint(checkpoint_path)
mpnn.eval()
print(mpnn)
test_path = r'data/smis.csv'
smiles_column = 'smiles'
df_test = pd.read_csv(test_path)

smis = df_test[smiles_column]

test_data = [data.MoleculeDatapoint.from_smi(smi) for smi in smis]
print(test_data[:5])
featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
test_dset = data.MoleculeDataset(test_data, featurizer=featurizer)
test_loader = data.build_dataloader(test_dset, shuffle=False)
with torch.no_grad():
    fingerprints = [
        mpnn.encoding(batch.bmg, batch.V_d, batch.X_d, i=0)
        for batch in test_loader
    ]
    fingerprints = torch.cat(fingerprints, 0)

print(fingerprints.shape)
print(fingerprints)

with torch.no_grad():
    encodings = [
        mpnn.encoding(batch.bmg, batch.V_d, batch.X_d, i=1)
        for batch in test_loader
    ]
    encodings = torch.cat(encodings, 0)

print(encodings.shape)
print(encodings)
