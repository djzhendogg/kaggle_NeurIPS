import pandas as pd


test_path = r'data/train.parquet'
df_test = pd.read_parquet(test_path)
df_test.drop_duplicates(subset=['molecule_smiles'], inplace=True)

print(df_test.shape)
