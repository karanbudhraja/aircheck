import joblib
import keras
from keras import metrics, models, layers, optimizers, losses
import numpy as np
import os
import pandas as pd
import pyarrow.parquet as pq
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import average_precision_score, precision_score, recall_score, log_loss

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
	ipca = IncrementalPCA(n_components=N_COMPONENTS, batch_size=subset_row_count, whiten=True)
	for (index, batch) in enumerate(parquet_file.iter_batches(batch_size=subset_row_count)):
		batch_data = batch.to_pandas()
		batch_data_x = np.hstack([np.vstack(batch_data[x]) for x in x_columns])

		# skip if insufficient rows
		if len(batch_data_x) < N_COMPONENTS:
			print(f"Skipping IPCA batch # {index+1} ({len(batch_data_x)} rows)")
			continue
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
print(f"Base occurrence rate is {base_occurrence_rate}")

#    
# Project data and train model
#

keras.utils.set_random_seed(RANDOM_STATE)

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
		# cast to float32 for speed and memory
		batch_data = batch.to_pandas()
		batch_data_x = np.hstack([np.vstack(batch_data[x]) for x in x_columns])
		batch_data_x = ipca.transform(batch_data_x).astype(np.float32)
		batch_data_y = batch_data[y_column]

		# use this data to train the model
		# class imbalance handling using weights
		history = model.fit(
			batch_data_x, 
			batch_data_y, 
			batch_size=256,
			class_weight={0: weight_non_occurrence, 1: weight_occurrence},
			verbose=0
			)
		current_metrics_data = {k: v[-1] for k, v in history.history.items()}
		
		loss = current_metrics_data["loss"]
		pr_auc = current_metrics_data["pr_auc"]
		precision = current_metrics_data["precision"]
		recall = current_metrics_data["recall"]
		print(f"Epoch {epoch+1} Batch {index+1} -> Loss: {loss:.4f} | PR-AUC: {pr_auc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

		epoch_batch_metrics_data.append(current_metrics_data)

	epoch_batch_metrics_data = pd.DataFrame(epoch_batch_metrics_data)
	epoch_batch_metrics_data.to_csv(f"mlp.epoch_batch_metrics_data.{epoch+1}.csv")

	# calculate global metrics for this epoch
	y_true, y_prob = [], []
	for (index, batch) in enumerate(parquet_file.iter_batches(batch_size=subset_row_count)):
		batch_data = batch.to_pandas()
		batch_data_x = np.hstack([np.vstack(batch_data[x]) for x in x_columns])
		batch_data_x = ipca.transform(batch_data_x).astype(np.float32)
		y_prob.append(model.predict(batch_data_x, batch_size=4096, verbose=0).ravel())
		y_true.append(batch_data[y_column].to_numpy())

	y_true = np.concatenate(y_true)
	y_prob = np.concatenate(y_prob)
	y_pred = (y_prob >= 0.5).astype(int)

	epoch_metrics = {
		"loss": log_loss(y_true, y_prob, labels=[0, 1]),
		"pr_auc": average_precision_score(y_true, y_prob),
		"precision": precision_score(y_true, y_pred, zero_division=0),
		"recall": recall_score(y_true, y_pred, zero_division=0),
	}
	print(f"Epoch {epoch+1} -> Loss: {epoch_metrics['loss']:.4f} | PR-AUC: {epoch_metrics['pr_auc']:.4f} | "
	      f"Precision: {epoch_metrics['precision']:.4f} | Recall: {epoch_metrics['recall']:.4f}")
	epoch_metrics_data.append(epoch_metrics)


				
	# ~ # calculate global metrics for this epoch
	# ~ model.reset_metrics()
	
	# ~ for (index, batch) in enumerate(parquet_file.iter_batches(batch_size=subset_row_count)):
		# ~ # cast to float32 for speed and memory
		# ~ batch_data = batch.to_pandas()
		# ~ batch_data_x = np.hstack([np.vstack(batch_data[x]) for x in x_columns])
		# ~ batch_data_x = ipca.transform(batch_data_x).astype(np.float32)
		# ~ batch_data_y = batch_data[y_column]
		# ~ model.test_on_batch(batch_data_x, batch_data_y)

	# ~ # get epoch metrics
	# ~ epoch_metrics = {k: float(v) for k, v in model.get_metrics_result().items()}
	# ~ loss = epoch_metrics["loss"]
	# ~ pr_auc = epoch_metrics["pr_auc"]
	# ~ precision = epoch_metrics["precision"]
	# ~ recall = epoch_metrics["recall"]
	# ~ print(f"Epoch {epoch+1} -> Loss: {loss:.4f} | PR-AUC: {pr_auc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

	# ~ epoch_metrics_data.append(epoch_metrics)

epoch_metrics_data = pd.DataFrame(epoch_metrics_data)
epoch_metrics_data.to_csv("mlp.epoch_metrics_data.csv")
