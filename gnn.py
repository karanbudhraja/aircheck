import joblib
import keras
from keras import metrics, models, layers, optimizers, losses
import numpy as np
import os
import pandas as pd
import pyarrow.parquet as pq
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import average_precision_score, precision_score, recall_score, log_loss

# ~ INPUT_DATA_PATH = "~/Downloads/DREAM_Challenge_1_TrainSet.parquet"
INPUT_DATA_PATH = "~/Downloads/DREAM_Challenge_1_TrainSet.subset.parquet"

DATA_SUBSET_FRACTION = 0.01
RANDOM_STATE = 0
N_EPOCHS = 30

is_use_cache = True
