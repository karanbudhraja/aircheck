import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import StratifiedGroupKFold

INPUT_DATA_PATH = "~/Downloads/DREAM_Challenge_1_TrainSet.parquet"
DATA_SUBSET_FRACTION = 0.01

#
# Load data
#

parquet_file = pq.ParquetFile(INPUT_DATA_PATH)
total_row_count = parquet_file.metadata.num_rows
subset_row_count = int(total_row_count * DATA_SUBSET_FRACTION)

print(f"Loading {subset_row_count}/{total_row_count} rows")
data = next(parquet_file.iter_batches(batch_size=subset_row_count))
data = data.to_pandas()

# extract scaffold
# only use valid rows
data_smiles = data["SMILES"]
data_scaffold = (
	# extract scaffold based on smiles
	data_smiles
	.map(lambda x: Chem.MolFromSmiles(x))
	.map(lambda x: MurckoScaffold.GetScaffoldForMol(x) if x else None)
	.map(lambda x: Chem.MolToSmiles(x) if x else None)
)
data = data[data_scaffold.notna()]

# columns subset
x_columns = ["AVALON", "FCFP4", "FCFP6"]
y_column = "LABEL"
data_x = np.hstack([np.vstack(data[x]) for x in x_columns])
data_y = data[y_column]

#
# Split data
#

# random seed for reproducibility
k_fold = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
for fold, (train_idx, test_idx) in enumerate(k_fold.split(data_x, data_y, groups=data_scaffold)):
	train_x, train_y = data_x[train_idx], data_y[train_idx]
	test_x, test_y = data_x[test_idx], data_y[test_idx]
	print(train_x.shape, train_y.shape)










