import numpy as np
import pandas as pd
import pyarrow.parquet as pq

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

# columns subset
print(data.columns)
x_columns = ["AVALON", "FCFP4", "FCFP6"]
y_column = "LABEL"
data = data[x_columns + [y_column]]
data_x = np.hstack([np.vstack(data[x]) for x in x_columns])
data_y = data[y_column]

print(data_x.shape)
