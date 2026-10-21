import joblib
import numpy as np
import os
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from ogb.utils import smiles2graph
from ogb.utils.features import get_atom_feature_dims
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import average_precision_score, precision_score, recall_score, log_loss
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool

# ~ INPUT_DATA_PATH = "~/Downloads/DREAM_Challenge_1_TrainSet.parquet"
INPUT_DATA_PATH = "~/Downloads/DREAM_Challenge_1_TrainSet.subset.parquet"

DATA_SUBSET_FRACTION = 0.01
RANDOM_STATE = 0
N_EPOCHS = 30
N_HIDDEN = 128
N_LAYERS = 3
DROPOUT = 0.2

RING_SIZES = [3, 4, 5, 6, 7, 8]

is_use_cache = True

#
# Load data
#

parquet_file = pq.ParquetFile(INPUT_DATA_PATH)
total_row_count = parquet_file.metadata.num_rows
subset_row_count = int(total_row_count * DATA_SUBSET_FRACTION)
print(f"Loading {subset_row_count}/{total_row_count} rows per batch")

x_column = "SMILES"
y_column = "LABEL"

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
# Graph parameters
#

def get_numeric_features(mol):
	mol_features = [[x.GetMass()] for x in mol.GetAtoms()]
	
	return mol_features
	
def get_categorical_features(mol):
    ring_information = mol.GetRingInfo()
    mol_features = []
    for atom in mol.GetAtoms():
        code = len(RING_SIZES)                   # default: other
        for (index, size) in enumerate(RING_SIZES):
            if ring_information.IsAtomInRingOfSize(atom.GetIdx(), size):
                code = index
                break
        mol_features.append([code])

    return mol_features

	
def smiles_to_graph_data(smiles_string, y):
	try:
		graph = smiles2graph(smiles_string)
		molecule = Chem.MolFromSmiles(smiles_string)
	except Exception as e:
		# invalid smiles
		return None
		
	graph_data = Data(
	# categorical features
	x_categorical = torch.cat([
		torch.tensor(graph["node_feat"], dtype=torch.long),
		torch.tensor(get_categorical_features(molecule), dtype=torch.long),
	], dim=1),

	# numeric features
	x_numeric=torch.tensor(scaler.transform(np.array(get_numeric_features(molecule))), dtype=torch.float),
	
	# metadata
	edge_index=torch.tensor(graph["edge_index"], dtype=torch.long),
	edge_attr=torch.tensor(graph["edge_feat"], dtype=torch.long),
	y=torch.tensor([y], dtype=torch.float),
	smiles=smiles_string,
	num_nodes=graph["num_nodes"],
	)
	
	return graph_data

# numeric feature scaling
save_path = "numeric_scaler.joblib"
if (is_use_cache == False) or (os.path.exists(save_path) == False):
    scaler = StandardScaler()
    for (index, batch) in enumerate(parquet_file.iter_batches(batch_size=subset_row_count)):
        batch_data = batch.to_pandas()
        for smiles_string in batch_data[x_column]:
            molecule = Chem.MolFromSmiles(smiles_string)
            if molecule is not None:
                scaler.partial_fit(np.array(get_numeric_features(molecule)))

        print(f"Processed scaler fit batch # {index+1}")

	# save to cache
    joblib.dump(scaler, save_path)

scaler = joblib.load(save_path)

# feature sizes
for (index, batch) in enumerate(parquet_file.iter_batches(batch_size=subset_row_count)):
	batch_data = batch.to_pandas()	
	batch_data = batch_data.apply(lambda x: smiles_to_graph_data(x[x_column], x[y_column]), axis=1)
	batch_data = batch_data[batch_data.notna()].tolist()
	
	# all nodes will have the same feature set
	break
	
categorical_dims = get_atom_feature_dims() + [len(RING_SIZES) + 1]
n_numeric = batch_data[0].x_numeric.size(1)

#
# Neural network structure
#

class NodeEncoder(nn.Module):
    def __init__(self, categorical_dims, n_numeric, n_hidden):
        super().__init__()
        self.categorical_embedding = nn.ModuleList(nn.Embedding(n_categories, n_hidden) for n_categories in categorical_dims)
        self.numeric_embedding = nn.Linear(n_numeric, n_hidden) if n_numeric > 0 else None

    def forward(self, x_categorical, x_numeric=None):
        h = sum(embedding(x_categorical[:, index]) for (index, embedding) in enumerate(self.categorical_embedding))
        if self.numeric_embedding is not None:
            h = h + self.numeric_embedding(x_numeric)
        return h
        
class MoleculeClassifier(nn.Module):
    def __init__(self, categorical_dims, n_numeric, n_hidden=N_HIDDEN, n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.node_encoder = NodeEncoder(categorical_dims, n_numeric, n_hidden)
        self.bond_encoders = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.bond_encoders.append(BondEncoder(n_hidden))
            mlp = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_hidden))
            self.convs.append(GINEConv(mlp))
            self.norms.append(nn.BatchNorm1d(n_hidden))
        self.head = nn.Sequential(
            nn.Linear(2 * n_hidden, n_hidden), nn.ReLU(), nn.Dropout(dropout), 
            nn.Linear(n_hidden, 1),                                            
        )
        self.dropout = dropout

    def forward(self, data):
        x = self.node_encoder(data.x_categorical, data.x_numeric)
        for conv, norm, bond_encoder in zip(self.convs, self.norms, self.bond_encoders):
            h = conv(x, data.edge_index, bond_encoder(data.edge_attr))
            
            # residual layer
            x = x + torch.relu(norm(h))

			# dropout layer
            x = nn.functional.dropout(x, self.dropout, self.training)
        g = torch.cat([global_add_pool(x, data.batch), global_mean_pool(x, data.batch)], dim=1)

        return self.head(g).squeeze(-1)

#
# Train model
#

torch.manual_seed(RANDOM_STATE)

# standard scikit-learn formula: weight = 1 / (num_classes * class_rate)
# manual limit based on square root of class imbalance
# ~ weight_occurrence = 1.0 / (2 * base_occurrence_rate)
# ~ weight_non_occurrence = 1.0 / (2 * (1-base_occurrence_rate))
weight_occurrence = min(1.0 / base_occurrence_rate, 30.0)
weight_non_occurrence = 1.0

# use bcewithlogit loss with linear output instead of bceloss with sigmoid output for stability
model = MoleculeClassifier(categorical_dims, n_numeric)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight_occurrence, dtype=torch.float))

def calculate_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "loss": log_loss(y_true, y_prob, labels=[0, 1]),
        "pr_auc": average_precision_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }

epoch_metrics_data = []
for epoch in range(N_EPOCHS):
	epoch_batch_metrics_data = []
	for (index, batch) in enumerate(parquet_file.iter_batches(batch_size=subset_row_count)):
		batch_data = batch.to_pandas()
		batch_data = batch_data.apply(lambda x: smiles_to_graph_data(x[x_column], x[y_column]), axis=1)
		batch_data = batch_data[batch_data.notna()].tolist()
		loader = DataLoader(batch_data, batch_size=256, shuffle=True)
		
		model.train()
		y_true, y_prob = [], []
		for graph_batch in loader:
			logits = model(graph_batch)
			loss = loss_fn(logits, graph_batch.y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			y_true.append(graph_batch.y.numpy())
			y_prob.append(torch.sigmoid(logits).detach().numpy())

		current_metrics_data = calculate_metrics(np.concatenate(y_true), np.concatenate(y_prob))

		loss = current_metrics_data["loss"]
		pr_auc = current_metrics_data["pr_auc"]
		precision = current_metrics_data["precision"]
		recall = current_metrics_data["recall"]
		print(f"Epoch {epoch+1} Batch {index+1} -> Loss: {loss:.4f} | PR-AUC: {pr_auc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

		epoch_batch_metrics_data.append(current_metrics_data)

	epoch_batch_metrics_data = pd.DataFrame(epoch_batch_metrics_data)
	epoch_batch_metrics_data.to_csv(f"gnn.epoch_batch_metrics_data.{epoch+1}.csv")
	
	# calculate global metrics for this epoch
	model.eval()
	y_true, y_prob = [], []
	with torch.no_grad():
		for (index, batch) in enumerate(parquet_file.iter_batches(batch_size=subset_row_count)):
			batch_data = batch.to_pandas()	
			batch_data = batch_data.apply(lambda x: smiles_to_graph_data(x[x_column], x[y_column]), axis=1)
			batch_data = batch_data[batch_data.notna()].tolist()
			loader = DataLoader(batch_data, batch_size=4096, shuffle=True)

			for graph_batch in loader:
				y_prob.append(torch.sigmoid(model(graph_batch)).numpy())
				y_true.append(graph_batch.y.numpy())

	epoch_metrics = calculate_metrics(np.concatenate(y_true), np.concatenate(y_prob))
	print(f"Epoch {epoch+1} -> Loss: {epoch_metrics['loss']:.4f} | PR-AUC: {epoch_metrics['pr_auc']:.4f} | "
	      f"Precision: {epoch_metrics['precision']:.4f} | Recall: {epoch_metrics['recall']:.4f}")
	epoch_metrics_data.append(epoch_metrics)

epoch_metrics_data = pd.DataFrame(epoch_metrics_data)
epoch_metrics_data.to_csv("gnn.epoch_metrics_data.csv")
