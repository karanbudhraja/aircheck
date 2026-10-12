import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

INPUT_DATA_PATH = "~/Downloads/DREAM_Challenge_1_TrainSet.parquet"
DATA_SUBSET_FRACTION = 0.01
RANDOM_STATE = 0

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

# dimensionality reduction
# scaling
data_processing_pipeline = Pipeline([
	('svd', TruncatedSVD(n_components=100, random_state=RANDOM_STATE)),
	('scaler', StandardScaler()),
])
data_processing_pipeline.fit(data_x)
data_x = data_processing_pipeline.transform(data_x)

#
# Split data
#

# random seed for reproducibility
k_fold = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
for fold, (train_idx, test_idx) in enumerate(k_fold.split(data_x, data_y, groups=data_scaffold)):
	# get data for this split
	train_x, train_y = data_x[train_idx], data_y[train_idx]
	test_x, test_y = data_x[test_idx], data_y[test_idx]

	# train model
	model = XGBClassifier(
		n_estimators=100,
		eval_metric='logloss',
		random_state=RANDOM_STATE,
	)
	model.fit(train_x, train_y)
	
	# evaluate model
	predicted_y = model.predict_proba(test_x)[:, 1]
	score = average_precision_score(test_y, predicted_y)
	print(score)


	break



