"""This script was used for RNA sequence prediction during the competition."""

import os
import torch
import pandas as pd
import numpy as np
from data import RNADatasetInference
from modeling_rna import ModelConfig, RNASeqPredictionModel
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter
from tqdm import tqdm
import matplotlib.pyplot as plt

import os
import pandas as pd

from tqdm import tqdm
from Bio import SeqIO


def read_fasta_biopython(file_path):
    sequences = {}
    for record in SeqIO.parse(file_path, "fasta-blast"):
        sequences[record.id] = str(record.seq)
    return sequences


rootpath = "./"

seq_list = os.listdir(f"{rootpath}/seqs")
content_dict = {"pdb_id": [], "seq": []}

for file in tqdm(seq_list):
    if 'fasta' in file:
        sequences = read_fasta_biopython(f"{rootpath}/seqs/{file}")
        content_dict["pdb_id"].append(list(sequences.keys())[0])
        content_dict["seq"].append(list(sequences.values())[0])
    # else:
    #     print(file)
data_df = pd.DataFrame(content_dict)


coord_dir = "./coords"

# Use the same configuration and parameters as in training.
test_dataset = RNADatasetInference(
    data_df,
    coord_dir,
    threshold=8,
    use_full_backbone=False,
    max_nucl_per_chunk=250,
    chunk_overlap=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
)

config = ModelConfig(
    coord_dim=3,
    num_atom_types=3,  # P, C4', N, set num_atom_types=7 if use_full_backbone
    node_dim=64,
    edge_dim=64,
    global_dim=64,
    num_layers=10,
    num_rbf=16,
    dropout_prob=0.2,
    vocab_size=5,  # A, U, C, G, [UNK]
    project_dim=1,
    num_prototypes=1,
) 
model = RNASeqPredictionModel(config)

model_path = "final_model.pt"
if torch.cuda.is_available():
    state_dict = torch.load(model_path)
else:
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

# test
def get_offset(local_nucl_ids: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """
    Get offset to turn local_nucl_ids into global nucl_ids in a batch.
    """
    num_nucl_per_rna = scatter(local_nucl_ids + 1, batch, dim=0, reduce="max")
    return torch.cumsum(
        torch.cat([local_nucl_ids.new_tensor([0]), num_nucl_per_rna]), dim=0
    )

token_to_word = {
    0: "A",
    1: "U",
    2: "C",
    3: "G",
    4: "X",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
content_dict = {"pdb_id": [], "seq": []}

recovery = []

with torch.no_grad():
    for data_batch in tqdm(test_loader):
        x = data_batch.x.to(device)
        edge_index = data_batch.edge_index.to(device)
        batch = data_batch.batch.to(device)
        edge_attr = data_batch.edge_attr.to(device)
        atom_ids = data_batch.atom_ids.to(device)
        seq = data_batch.seq.to(device)
        local_nucl_ids = data_batch.nucl_ids.to(device)
        nucl_offsets = get_offset(local_nucl_ids, batch)
        batch_nucl_ids = local_nucl_ids + nucl_offsets[batch]
        batch_nucl_ids = batch_nucl_ids.to(device)

        logits = model(
            x, edge_attr, edge_index, batch, atom_ids, batch_nucl_ids
        )

        pred_seq = torch.argmax(logits, dim=-1)
        pred_seq = pred_seq.cpu().numpy()
        recovery.append((pred_seq == seq.cpu().numpy()).sum() / len(pred_seq))
        pred_seq = "".join([token_to_word[i] for i in pred_seq])

        pdb_id = data_batch.pdb_id[0]
        content_dict["pdb_id"].append(pdb_id[0])
        content_dict["seq"].append(pred_seq)

print("recovery: ", np.mean(recovery))

# save to csv
pred_df = pd.DataFrame(content_dict)
if not os.path.exists("./result"):
    os.makedirs("./result", exist_ok=True)
pred_df.to_csv("./result/submit.csv", index=False)

