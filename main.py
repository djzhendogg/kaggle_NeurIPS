import pandas as pd
import torch
import numpy as np
from joblib import Parallel, delayed

from chemprop import data, featurizers, models

checkpoint_path = r'encoder/example_model_v2_regression_mol.ckpt'
mpnn = models.MPNN.load_from_checkpoint(checkpoint_path)
mpnn.eval()

test_path = r'data/train.parquet'
smiles_column = 'molecule_smiles'
df_test = pd.read_parquet(test_path)
df_test = df_test.sample(n=100, random_state=1)
several_id_lists = np.array_split(df_test.to_numpy(), 3)

def encode(smis):
    test_data = [data.MoleculeDatapoint.from_smi(smi) for smi in smis]
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    test_dset = data.MoleculeDataset(test_data, featurizer=featurizer)
    test_loader = data.build_dataloader(test_dset, shuffle=False)

    with torch.no_grad():
        encodings = [
            mpnn.encoding(batch.bmg, batch.V_d, batch.X_d, i=1)
            for batch in test_loader
        ]
        encodings = torch.cat(encodings, 0)
    return encodings

def replace_prot(prot_col):
    labeled = []
    for i in prot_col:
        if i == 'BRD4':
            labeled.append(1)
        elif i == 'HSA':
            labeled.append(2)
        elif i == 'sEH':
            labeled.append(3)
        else:
            labeled.append(4)
    return labeled

def to_l_space(df):
    id_1 = df[:, 0][0]
    print(id_1)
    id_last = df[:, 0][-1]
    print(id_last)

    smis = df[:, 4]
    print(smis[0])
    mol_space = encode(smis)
    print(mol_space[0])

    prot_col = df[:, 5]
    print(prot_col[0])
    prot_space = replace_prot(prot_col)
    print(prot_space[0])

    fin_data = pd.DataFrame(mol_space.numpy())
    fin_data['protein_name'] = prot_space
    fin_data['target'] = df[:, 6]
    print(fin_data[:5])
    fin_data.to_parquet(f'data/lspace/ls_{id_1}_{id_last}.parquet')

for i in several_id_lists:
    to_l_space(i)

# func_out = Parallel(n_jobs=-1)(
#     [
#         delayed(to_l_space)(
#             length
#         ) for length in several_id_lists
#     ]
# )