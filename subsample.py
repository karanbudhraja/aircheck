import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

N_NEGATIVES = 100_000
INPUT_DATA_PATH = "~/Downloads/DREAM_Challenge_1_TrainSet.parquet"
DATA_SUBSET_FRACTION = 0.01
RANDOM_STATE = 0

y_column = "LABEL"

#
# batch size calculation
#

parquet_file = pq.ParquetFile(INPUT_DATA_PATH)
total_row_count = parquet_file.metadata.num_rows
subset_row_count = int(total_row_count * DATA_SUBSET_FRACTION)
print(f"Loading {subset_row_count}/{total_row_count} rows per batch")

#
# total negative count
#

n_negatives_total = 0
n_positives_total = 0
for (index, batch) in enumerate(parquet_file.iter_batches(batch_size=subset_row_count)):
	batch_data = batch.to_pandas()
	batch_data_y = batch_data[y_column]
	n_positives_total += batch_data_y.sum()
	n_negatives_total += (1-batch_data_y).sum()

print(f"Total negative {n_negatives_total}, total positive {n_positives_total}")

#
# sampling
#

keep_prob = N_NEGATIVES / n_negatives_total
print(f"Downsampling to {N_NEGATIVES}/{n_negatives_total} negatives")

rng = np.random.default_rng(RANDOM_STATE)
save_path = "~/Downloads/DREAM_Challenge_1_TrainSet.subset.parquet"

writer = None
n_kept = 0

for batch in parquet_file.iter_batches(batch_size=subset_row_count):
    y = batch.column(y_column).to_numpy()
    keep = (y == 1) | (rng.random(len(y)) < keep_prob)
    if not keep.any():
        continue
    kept = pa.Table.from_batches([batch.filter(pa.array(keep))])
    if writer is None:
        writer = pq.ParquetWriter(save_path, kept.schema)
    writer.write_table(kept)
    n_kept += kept.num_rows

writer.close()
print(f"Saved {n_kept} rows to {save_path}")

# total negative count
parquet_file = pq.ParquetFile(save_path)
n_negatives_total = 0
n_positives_total = 0
for (index, batch) in enumerate(parquet_file.iter_batches(batch_size=subset_row_count)):
	batch_data = batch.to_pandas()
	batch_data_y = batch_data[y_column]
	n_positives_total += batch_data_y.sum()
	n_negatives_total += (1-batch_data_y).sum()

print(f"Total subsampled negative {n_negatives_total}, total subsampled positive {n_positives_total}")
