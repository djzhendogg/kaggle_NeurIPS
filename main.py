import pandas as pd
import torch
import numpy as np
# from joblib import Parallel, delayed

from chemprop import data, featurizers, models

checkpoint_path = r'encoder/example_model_v2_regression_mol.ckpt'
mpnn = models.MPNN.load_from_checkpoint(checkpoint_path)
mpnn.eval()
print(f"start read")
test_path = r'data/train.parquet'
smiles_column = 'molecule_smiles'
df_test = pd.read_parquet(test_path)
print(f"read done")
print(f"drop_duplicates start")
df_test.drop_duplicates(subset=['molecule_smiles'], inplace=True)
print(f"drop_duplicates done")
print(f"split start")
df_test = df_test[0:100000]
several_id_lists = np.array_split(df_test.to_numpy(), 40)
print(f"split done")


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
    fin_df = pd.DataFrame()
    id_1 = df[:, 0][0]
    id_last = df[:, 0][-1]
    print(f"start {id_1} {id_last}")
    chancks = np.array_split(df, 1000)
    n = 0
    for chank in chancks:
        n += 1
        smis = chank[:, 4]
        mol_space = encode(smis)
        del smis
        prot_col = chank[:, 5]
        prot_space = replace_prot(prot_col)

        fin_data = pd.DataFrame(mol_space.numpy())
        fin_data['protein_name'] = prot_space
        fin_data['target'] = chank[:, 6]
        fin_data['id'] = chank[:, 0]
        fin_df = pd.concat([fin_df, fin_data])
        del fin_data
        print(f"append {n}")

    print(f"saving {id_1} {id_last}")
    fin_df.to_parquet(f'data/lspace/ls_{id_1}_{id_last}.parquet')


for mol_list in several_id_lists:
    to_l_space(mol_list)
# print(f"func start")
# func_out = Parallel(n_jobs=48)(
#     [
#         delayed(to_l_space)(
#             mol_list
#         ) for mol_list in several_id_lists
#     ]
# )
