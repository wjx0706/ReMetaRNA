import torch
from data import RNADataset, RNADatasetChunked, ContrastiveDataset, AugmentTransform
from modeling_rna import ModelConfig, RNAMultitaskModel
from trainer import TrainingArguments, Trainer


# Run preprocessing.py to generate the required CSV files before training.
train_data_path = "./public_train_data.csv"
eval_data_path = "./public_valid_data.csv"
coord_dir = "./coords/"


train_dataset = RNADatasetChunked(
    train_data_path,
    coord_dir,
    threshold=8,
    use_full_backbone=False,
    max_nucl_per_chunk=250,
    chunk_overlap=125,
)
eval_dataset = RNADatasetChunked(
    eval_data_path,
    coord_dir,
    threshold=8,
    use_full_backbone=False,
    max_nucl_per_chunk=250,
    chunk_overlap=125,
)


transform = AugmentTransform()
train_dataset = ContrastiveDataset(train_dataset, transform=transform)
eval_dataset = ContrastiveDataset(eval_dataset, transform=transform)


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

model = RNAMultitaskModel(config)


lr = 0.29
training_args = TrainingArguments(
    num_epochs=100,
    training_batch_size=16,
    eval_batch_size=16,
    optimizer_type="sgd",
    learning_rate=lr,
    warmup_init_lr=0.01 * lr,
    min_lr=1e-4 * lr,
    weight_decay=1e-5,
    use_cos_scheduler=True,
    warmup_ratio=0.1,
    temperature=0.25,
    use_amp=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_best_model=True,
    seed=0,
    sinkhorn_eps=0.03,
    freeze_prototypes_nepochs=1,
)


trainer = Trainer(
    model=model,
    training_args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)


trainer.train()
