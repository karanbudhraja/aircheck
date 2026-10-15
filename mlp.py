import joblib
from keras import metrics, models, layers, optimizers, losses
import numpy as np
import os
import pandas as pd
import pyarrow.parquet as pq
from sklearn.decomposition import IncrementalPCA

INPUT_DATA_PATH = "~/Downloads/DREAM_Challenge_1_TrainSet.parquet"
DATA_SUBSET_FRACTION = 0.01
N_COMPONENTS = 1000
RANDOM_STATE = 0
N_EPOCHS = 10

is_use_cache = True

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

save_path = f"ipca.{N_COMPONENTS}.joblib"
if (is_use_cache == False) or (os.path.exists(save_path) == False):
	ipca = IncrementalPCA(n_components=N_COMPONENTS, batch_size=subset_row_count)
	for (index, batch) in enumerate(parquet_file.iter_batches(batch_size=subset_row_count)):
		batch_data = batch.to_pandas()
		batch_data_x = np.hstack([np.vstack(batch_data[x]) for x in x_columns])
		ipca.partial_fit(batch_data_x)
	  
		print(f"Processed IPCA fit batch # {index+1}")
		
	# save to cache
	joblib.dump(ipca, save_path)

ipca = joblib.load(save_path)

#
# Base occurrence rate
#

save_path = f"base_occurrence_rate.joblib"
if (is_use_cache == False) or (os.path.exists(save_path) == False):
	occurrence_count = 0
	total_count = 0
	for (index, batch) in enumerate(parquet_file.iter_batches(batch_size=subset_row_count)):
		batch_data = batch.to_pandas()
		batch_data_y = batch_data[y_column]
		occurrence_count += batch_data_y.sum()
		total_count += len(batch_data_y)

		print(f"Processed base occurrence batch # {index+1}")

	base_occurrence_rate = occurrence_count / total_count

	# save to cache
	joblib.dump(base_occurrence_rate, save_path)

base_occurrence_rate = joblib.load(save_path)

#    
# Project data and train model
#

# standard scikit-learn formula: weight = 1 / (num_classes * class_rate)
weight_occurrence = 1.0 / (2 * base_occurrence_rate)
weight_non_occurrence = 1.0 / (2 * (1-base_occurrence_rate))

model = models.Sequential([
	layers.Input(shape=(N_COMPONENTS,)),
	layers.Dense(256, activation='relu'),
	layers.Dropout(0.3),
	layers.Dense(64, activation='relu'),
	layers.Dropout(0.2),
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

epoch_metrics_data = []
for epoch in range(N_EPOCHS):
	epoch_batch_metrics_data = []
	for (index, batch) in enumerate(parquet_file.iter_batches(batch_size=subset_row_count)):
		batch_data = batch.to_pandas()

		# shuffle data
		batch_data = batch_data.sample(frac=1, random_state=RANDOM_STATE)

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
		
		loss, pr_auc, precision, recall = metrics_output
		print(f"Epoch {epoch+1} Batch {index+1} -> Loss: {loss:.4f} | PR-AUC: {pr_auc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

		current_metrics_data = {
		"loss": loss,
		"pr_auc": pr_auc,
		"precision": precision,
		"recall": recall,
		}
		epoch_batch_metrics_data.append(current_metrics_data)

	epoch_batch_metrics_data = pd.DataFrame(epoch_batch_metrics_data)
	epoch_batch_metrics_data.to_csv(f"mlp.epoch_batch_metrics_data.{epoch+1}.csv")
				
	# calculate global metrics for this epoch
	model.reset_metrics()
	
	epoch_loss = 0
	total_processed_rows = 0
	for (index, batch) in enumerate(parquet_file.iter_batches(batch_size=subset_row_count)):
		batch_data = batch.to_pandas()
		batch_data_x = np.hstack([np.vstack(batch_data[x]) for x in x_columns])
		batch_data_x = ipca.transform(batch_data_x)
		batch_data_y = batch_data[y_column]
		
		metrics_output = model.test_on_batch(batch_data_x, batch_data_y)
		loss, pr_auc, precision, recall = metrics_output
		epoch_loss += loss * len(batch_data_y)
		total_processed_rows += len(batch_data_y)
			
	# calculate weighted average
	epoch_loss /= total_processed_rows

	# get epoch metrics
	epoch_metrics = model.get_metrics_result()
	pr_auc = epoch_metrics["pr_auc"]
	precision = epoch_metrics["precision"]
	recall = epoch_metrics["recall"]
	print(f"Epoch {epoch+1} -> Loss: {epoch_loss:.4f} | PR-AUC: {pr_auc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

	epoch_metrics["loss"] = epoch_loss
	epoch_metrics_data.append(epoch_metrics)

epoch_metrics_data = pd.DataFrame(epoch_metrics_data)
epoch_metrics_data.to_csv("mlp.epoch_metrics_data.csv")

# ~ # save model
# ~ model.save("mlp.keras")
