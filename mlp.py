from keras import metrics, models, layers, optimizers, losses
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.decomposition import IncrementalPCA

INPUT_DATA_PATH = "~/Downloads/DREAM_Challenge_1_TrainSet.parquet"
DATA_SUBSET_FRACTION = 0.001
N_COMPONENTS = 100
RANDOM_STATE = 0

#
# Load data
#

parquet_file = pq.ParquetFile(INPUT_DATA_PATH)
total_row_count = parquet_file.metadata.num_rows
subset_row_count = int(total_row_count * DATA_SUBSET_FRACTION)
print(f"Loading {subset_row_count}/{total_row_count} rows per batch")

x_columns = ["AVALON", "FCFP4", "FCFP6"]
y_column = "LABEL"

#
# Train data projection
#

ipca = IncrementalPCA(n_components=N_COMPONENTS, batch_size=subset_row_count)
for (index, batch) in enumerate(parquet_file.iter_batches(batch_size=subset_row_count)):
	batch_data = batch.to_pandas()
	batch_data_x = np.hstack([np.vstack(batch_data[x]) for x in x_columns])
	ipca.partial_fit(batch_data_x)
  
	print(f"Processed IPCA fit batch # {index+1}")
	
	# ~ # TODO: REMOVE FOR FULL DATA RUN
	# ~ if(index == 1):
		# ~ break

#
# Base occurrence rate
#

occurrence_count = 0
total_count = 0
for (index, batch) in enumerate(parquet_file.iter_batches(batch_size=subset_row_count)):
	batch_data = batch.to_pandas()
	batch_data_y = batch_data[y_column]
	occurrence_count += batch_data_y.sum()
	total_count += len(batch_data_y)

	print(f"Processed base occurrence batch # {index+1}")
	
	# ~ # TODO: REMOVE FOR FULL DATA RUN
	# ~ if(index == 10):
		# ~ break

base_occurrence_rate = occurrence_count / total_count

#    
# Project data and train model
#

# standard scikit-learn formula: weight = 1 / (num_classes * class_rate)
weight_occurrence = 1.0 / (2 * base_occurrence_rate)
weight_non_occurrence = 1.0 / (2 * (1-base_occurrence_rate))

model = models.Sequential([
	layers.Input(shape=(N_COMPONENTS,)),
	layers.Dense(64, activation='relu'),
	layers.Dropout(0.3),
	layers.Dense(1, activation='sigmoid')
	])
	
model_metrics = [
    metrics.AUC(curve='PR', name='pr_auc'),
    metrics.Precision(name='precision'),
    metrics.Recall(name='recall')
]
model.compile(
	optimizer=optimizers.Adam(learning_rate=0.001),
	loss=losses.BinaryCrossentropy(),
	metrics=model_metrics
	)

for (index, batch) in enumerate(parquet_file.iter_batches(batch_size=subset_row_count)):
	batch_data = batch.to_pandas()
	batch_data_x = np.hstack([np.vstack(batch_data[x]) for x in x_columns])
	batch_data_x = ipca.transform(batch_data_x)
	batch_data_y = batch_data[y_column]

	print(f"Applied IPCA to data batch # {index+1}")

	# use this data to train the model
	# class imbalance handling using weights
	metrics_output = model.train_on_batch(
        batch_data_x, 
        batch_data_y,
        class_weight={0: weight_non_occurrence, 1: weight_occurrence}
    )
	
	x = 1
	loss, pr_auc, precision, recall = metrics_output
	print(f"Batch {index+1} -> Loss: {loss:.4f} | PR-AUC: {pr_auc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    
	# ~ # TODO: REMOVE FOR FULL DATA RUN
	# ~ if(index == 1):
		# ~ break

# save model
model.save("mlp.keras")
