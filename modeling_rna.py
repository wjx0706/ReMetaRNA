from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphNorm, LayerNorm
from torch_geometric.nn.models import MetaLayer
from torch_geometric.utils import scatter


@dataclass
class ModelConfig:
    """
    Configuration class for RNAStructFormer model parameters.

    Attributes:
        coord_dim: Dimension of coordinate features (default: 3 for 3D coordinates)
        num_atom_types: Number of atom types in the RNA backbone (default: 7)
        node_dim: Dimension of node features (default: 128)
        edge_dim: Dimension of edge features (default: 128)
        global_dim: Dimension of global features (default: 128)
        num_layers: Number of GNN layers (default: 4)
        num_rbf: Number of radial basis functions for distance encoding (default: 32)
        dropout_prob: Dropout probability (default: 0.1)
        vocab_size: Size of nucleotide vocabulary (default: 5 for A, U, C, G, [UNK])
        project_dim: Dimension for contrastive learning projection (default: 128)
        num_prototypes: Number of prototypes for contrastive learning (default: 100)
    """

    # Basic dimensions
    coord_dim: int = 3
    num_atom_types: int = (
        7  # P,O5',C5',C4',C3',O3',N(N1/N9) or 3 when use_full_backbone=False
    )
    node_dim: int = 128
    edge_dim: int = 128
    global_dim: int = 128

    # Model architecture
    num_layers: int = 4
    num_rbf: int = 32
    dropout_prob: float = 0.1

    # For prediction head
    vocab_size: int = 5  # A, U, C, G, [UNK]

    # For contrastive learning
    project_dim: int = 128
    num_prototypes: int = 100


class NodeInitLayer(nn.Module):
    """
    Node initialization layer for the RNAStructFormer model.
        coord -> linear(coord)
        atom_type -> embedding
    """

    def __init__(
        self,
        coord_dim: int,
        num_atom_types: int,
        node_dim: int,
        dropout_prob: float,
    ) -> None:
        super().__init__()
        self.coord_linear = nn.Linear(coord_dim, node_dim)  # [3, node_dim]
        self.atom_embedding = nn.Embedding(num_atom_types, node_dim)  # [7, node_dim]
        # self.dropout = nn.Dropout(dropout_prob)
        # self.coord_norm = GraphNorm(coord_dim)
        self.node_norm = GraphNorm(node_dim)

    def forward(
        self, x: torch.Tensor, atom_ids: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        # x: [num_atoms, 3]
        # atom_ids: [num_atoms]

        # x = self.coord_norm(x, batch=batch)
        x = F.relu(self.coord_linear(x))
        atom_embedding = self.atom_embedding(atom_ids)

        return self.node_norm(x + atom_embedding, batch=batch)


class RBFEncoder(nn.Module):
    """
    Radial basis function encoder for distance features.

    Transforms scalar distance values into high-dimensional vectors using a
    set of Gaussian basis functions with fixed centers evenly spaced between
    rbf_min and rbf_max.
    """

    def __init__(
        self, num_rbf: int, rbf_min: float = 0.0, rbf_max: float = 12.0
    ) -> None:
        super().__init__()
        self.mu: torch.Tensor
        self.register_buffer(
            "mu", torch.linspace(rbf_min, rbf_max, num_rbf)
        )  # [num_rbf]
        self.gamma = 1.0 / ((rbf_max - rbf_min) / num_rbf) ** 2  # inverse sigma^2

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        d_expand = distances.unsqueeze(-1)  # [num_edges, 1]
        rbf = torch.exp(-self.gamma * (d_expand - self.mu) ** 2)  # broadcasting

        return rbf


class EdgeInitLayer(nn.Module):
    """
    Edge initialization layer for the RNAStructFormer model.

    Creates initial edge representations by combining:
    1. The average of connected node features
    2. RBF encoding of the distance between nodes
    """

    def __init__(self, node_dim: int, edge_dim: int, num_rbf: int) -> None:
        super().__init__()
        self.rbf_linear = nn.Linear(
            num_rbf, edge_dim, bias=False
        )  # no bias, just change dim
        self.edge_linear = nn.Linear(node_dim, edge_dim)
        # self.edge_norm = GraphNorm(edge_dim)
        self.rbf_encoder = RBFEncoder(num_rbf=num_rbf)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,  # , batch: torch.Tensor
    ) -> torch.Tensor:
        # x: [num_atoms, node_dim]
        # edge_index: [2, num_edges]
        # edge_attr: [num_edges]
        # batch: [num_edges] with max entry B - 1.

        src, dst = edge_index
        aggr_node_attr = (x[src] + x[dst]) / 2

        edge_attr = self.rbf_encoder(edge_attr)
        edge_attr = self.rbf_linear(edge_attr)
        edge_attr += self.edge_linear(aggr_node_attr)
        # edge_attr = self.edge_norm(edge_attr, batch=batch)

        return edge_attr


class GlobalInitLayer(nn.Module):
    """
    Global initialization layer for the RNAStructFormer model.
    Initializes the global features as the average of the edge features.
    """

    def __init__(self, edge_dim: int, global_dim: int) -> None:
        super().__init__()
        self.global_linear = nn.Linear(edge_dim, global_dim)
        self.norm = LayerNorm(global_dim, mode="node")

    def forward(self, edge_attr: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        u = scatter(edge_attr, batch, dim=0, reduce="mean")
        u = self.global_linear(u)

        return self.norm(u)


class InitLayer(nn.Module):
    """
    Initialization layer for the RNAStructFormer model.
    Combines node, edge, and global initialization layers.
    """

    def __init__(
        self,
        coord_dim: int,
        num_atom_types: int,
        node_dim: int,
        edge_dim: int,
        global_dim: int,
        num_rbf: int,
        dropout_prob: float,
    ) -> None:
        super().__init__()
        self.node_init = NodeInitLayer(
            coord_dim, num_atom_types, node_dim, dropout_prob
        )
        self.edge_init = EdgeInitLayer(node_dim, edge_dim, num_rbf=num_rbf)
        self.global_init = GlobalInitLayer(edge_dim, global_dim)

    def forward(
        self,
        x: torch.Tensor,
        atom_ids: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.node_init(x, atom_ids, batch)
        # edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=0.0)
        batch_for_edge = batch[edge_index[0]]
        edge_attr = self.edge_init(x, edge_index, edge_attr)
        u = self.global_init(edge_attr, batch=batch_for_edge)

        return x, edge_attr, u, edge_index


class EdgeUpdateLayer(nn.Module):
    """
    Edge update layer for the RNAStructFormer model.

    This layer updates edge attributes by combining information from:
    1. Source node features
    2. Destination node features
    3. Current edge attributes
    4. Global graph features
    """

    def __init__(
        self, node_dim: int, edge_dim: int, global_dim: int, dropout_prob: float
    ) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim + global_dim, 4 * edge_dim),
            nn.ReLU(inplace=True),
            nn.Linear(4 * edge_dim, edge_dim),
        )
        self.edge_norm = GraphNorm(edge_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_attr: torch.Tensor,
        u: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        # src, dst: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.

        out = torch.cat([src, dst, edge_attr, u[batch]], dim=1)
        out = self.edge_mlp(out)
        out = self.dropout(out)

        return self.edge_norm(out + edge_attr, batch=batch)


class NodeUpdateLayer(nn.Module):
    """
    Node update layer for the RNAStructFormer model.

    This layer updates node attributes through a two-step process:
    1. Message aggregation: Collect and transform messages from neighboring edges
    2. Node update: Combine node features with aggregated messages and global context
    """

    def __init__(
        self, node_dim: int, edge_dim: int, global_dim: int, dropout_prob: float
    ) -> None:
        super().__init__()
        self.node_aggregate_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, 4 * edge_dim),
            nn.ReLU(inplace=True),
            nn.Linear(4 * edge_dim, node_dim),
        )
        self.node_update_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim + global_dim, 4 * node_dim),
            nn.ReLU(inplace=True),
            nn.Linear(4 * node_dim, node_dim),
        )

        self.node_norm = GraphNorm(node_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        u: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [N] with max entry B - 1.
        # src, dst: [E]

        # 1. Aggregate edge attributes per node
        src, dst = edge_index
        agg = torch.cat([x[src], edge_attr], dim=1)  # [E, F_x + F_e]
        agg = self.node_aggregate_mlp(agg)
        # TODO: introduce attention
        agg = scatter(agg, dst, dim=0, dim_size=x.size(0), reduce="mean")

        # 2. Compute updated node attributes
        out = torch.cat([x, agg, u[batch]], dim=1)
        out = self.node_update_mlp(out)
        out = self.dropout(out)

        return self.node_norm(out + x, batch=batch)


class GlobalUpdateLayer(nn.Module):
    """
    Global update layer for the RNAStructFormer model.

    This layer updates global graph attributes by aggregating information from
    all nodes and edges in the graph.
    """

    def __init__(
        self, global_dim: int, node_dim: int, edge_dim: int, dropout_prob: float
    ) -> None:
        super().__init__()
        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim + node_dim + edge_dim, 4 * global_dim),
            nn.ReLU(inplace=True),
            nn.Linear(4 * global_dim, global_dim),
        )
        self.global_norm = LayerNorm(global_dim, mode="node")
        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        u: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [N] with max entry B - 1.

        # alternative approach:
        # 1. Aggregate per nucleotide
        # 2. Build graph on nucleotide level
        # 3. TransformerConv on nucleotide level
        # 4. update u using pooling of nucleotide level features

        out = torch.cat(
            [
                u,
                scatter(edge_attr, batch[edge_index[0]], dim=0, reduce="mean"),
                scatter(x, batch, dim=0, reduce="mean"),
            ],
            dim=1,
        )

        out = self.global_mlp(out)
        out = self.dropout(out)

        return self.global_norm(out + u)


class PredictionHead(nn.Module):
    def __init__(self, global_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(global_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ContrastiveProjector(nn.Module):
    def __init__(self, global_dim: int, project_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(global_dim, 4 * project_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(4 * project_dim, project_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class RNAStructFormer(nn.Module):
    """
    Graph Neural Network for RNA structure modeling.

    Processes RNA structure graphs through:
    1. Initial feature encoding of atoms, bonds, and global features
    2. Multiple message-passing layers to refine representations
    3. Output representations at atom, nucleotide, and global levels
    """

    def __init__(
        self,
        num_atom_types: int,
        coord_dim: int,
        node_dim: int,
        edge_dim: int,
        global_dim: int,
        num_layers: int,
        num_rbf: int,
        dropout_prob: float,
    ) -> None:
        super().__init__()
        self.init_layer = InitLayer(
            coord_dim=coord_dim,
            num_atom_types=num_atom_types,
            node_dim=node_dim,
            edge_dim=edge_dim,
            global_dim=global_dim,
            num_rbf=num_rbf,
            dropout_prob=dropout_prob,
        )
        # self.num_atom_types = num_atom_types

        # stack layers
        self.layers = nn.ModuleList(
            [
                self.get_metalayer(node_dim, edge_dim, global_dim, dropout_prob)
                for _ in range(num_layers)
            ]
        )

        # self.prediction_head = PredictionHead(global_dim, vocab_size)

    def get_metalayer(
        self, node_dim: int, edge_dim: int, global_dim: int, dropout_prob: float
    ) -> nn.Module:
        edge_update = EdgeUpdateLayer(
            node_dim=node_dim,
            edge_dim=edge_dim,
            global_dim=global_dim,
            dropout_prob=dropout_prob,
        )
        node_update = NodeUpdateLayer(
            node_dim=node_dim,
            edge_dim=edge_dim,
            global_dim=global_dim,
            dropout_prob=dropout_prob,
        )
        global_update = GlobalUpdateLayer(
            global_dim=global_dim,
            node_dim=node_dim,
            edge_dim=edge_dim,
            dropout_prob=dropout_prob,
        )

        return MetaLayer(
            edge_model=edge_update,
            node_model=node_update,
            global_model=global_update,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        atom_ids: torch.Tensor,
        nucl_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [num_atoms, 3]
        # edge_index: [2, num_edges]
        # edge_attr: [num_edges]
        # atom_ids: [num_atoms]
        # batch: [num_atoms] with max entry B - 1.

        x, edge_attr, u, edge_index = self.init_layer(
            x=x,
            atom_ids=atom_ids,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
        )

        for layer in self.layers:
            x, edge_attr, u = layer(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                u=u,
                batch=batch,
            )

        # get nucleotide by mean pooling
        nucl_repr = scatter(x, nucl_ids, dim=0, reduce="mean")

        return x, edge_attr, u, nucl_repr


class RNAMultitaskModel(nn.Module):
    """
    Multi-task learning model for RNA sequence and structure analysis.

    Extends RNAStructFormer to perform multiple tasks:
    1. Nucleotide sequence prediction
    2. Contrastive learning for structure representation

    The model processes both the original structure and an augmented version
    simultaneously for contrastive learning objectives.
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.backbone = RNAStructFormer(
            model_config.num_atom_types,
            model_config.coord_dim,
            model_config.node_dim,
            model_config.edge_dim,
            model_config.global_dim,
            model_config.num_layers,
            model_config.num_rbf,
            model_config.dropout_prob,
        )

        # projector for contrastive learning
        self.contrastive_projector = ContrastiveProjector(
            model_config.global_dim, model_config.project_dim
        )
        self.prototypes = nn.Linear(
            model_config.project_dim, model_config.num_prototypes, bias=False
        )

        # prediction head
        self.prediction_head = PredictionHead(
            model_config.node_dim, model_config.vocab_size
        )

        self.reset_parameters()

    def forward(
        self,
        x_s: torch.Tensor,
        x_t: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        atom_ids: torch.Tensor,
        nucl_ids: torch.Tensor,
    ):
        num_nodes = x_s.size(0)  # num of nodes
        batch_offset = (
            int(batch.max()) + 1
        )  # num of sub graphs == batch size in dataloader
        nucl_id_offset = int(nucl_ids.max()) + 1  # num of nucleotides

        # forward pass for original + augmented graph
        _, _, u, nucl_repr = self.backbone(
            x=torch.cat([x_s, x_t], dim=0),
            edge_attr=edge_attr.repeat(2),
            edge_index=torch.cat([edge_index, edge_index + num_nodes], dim=1),
            batch=torch.cat([batch, batch + batch_offset], dim=0),
            atom_ids=atom_ids.repeat(2),
            nucl_ids=torch.cat([nucl_ids, nucl_ids + nucl_id_offset], dim=0),
        )

        # predict sequence
        logits = self.prediction_head(nucl_repr)

        # swap prediction
        z = self.contrastive_projector(u)
        z = z / z.norm(dim=1, keepdim=True, p=2)

        scores = self.prototypes(z)  # [2 * num_nodes, num_prototypes]
        scores_s, scores_t = scores[:batch_offset], scores[batch_offset:]

        return logits, scores_s, scores_t

    def reset_parameters(self) -> None:
        self.apply(self.init_weights)

    def init_weights(self, m: nn.Module) -> None:
        if m is self.prototypes:
            with torch.no_grad():
                weight = cast(nn.Parameter, m.weight)
                weight.copy_(torch.randn_like(weight))
                weight.div_(weight.norm(dim=1, keepdim=True))
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, (GraphNorm, LayerNorm)):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)


class RNASeqPredictionModel(nn.Module):
    """
    Sequence prediction model for RNA sequence and structure analysis.

    Extends RNAStructFormer to perform multiple tasks:
    1. Nucleotide sequence prediction
    2. Contrastive learning for structure representation

    The model processes both the original structure and an augmented version
    simultaneously for contrastive learning objectives.
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.backbone = RNAStructFormer(
            model_config.num_atom_types,
            model_config.coord_dim,
            model_config.node_dim,
            model_config.edge_dim,
            model_config.global_dim,
            model_config.num_layers,
            model_config.num_rbf,
            model_config.dropout_prob,
        )

        # projector for contrastive learning
        self.contrastive_projector = ContrastiveProjector(
            model_config.global_dim, model_config.project_dim
        )
        self.prototypes = nn.Linear(
            model_config.project_dim, model_config.num_prototypes, bias=False
        )

        # prediction head
        self.prediction_head = PredictionHead(
            model_config.node_dim, model_config.vocab_size
        )

        self.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        atom_ids: torch.Tensor,
        nucl_ids: torch.Tensor,
    ):
        # forward pass for original
        _, _, _, nucl_repr = self.backbone(
            x=x,
            edge_attr=edge_attr,
            edge_index=edge_index,
            batch=batch,
            atom_ids=atom_ids,
            nucl_ids=nucl_ids,
        )

        # predict sequence
        logits = self.prediction_head(nucl_repr)

        return logits

    def reset_parameters(self) -> None:
        self.apply(self.init_weights)

    def init_weights(self, m: nn.Module) -> None:
        if m is self.prototypes:
            with torch.no_grad():
                weight = cast(nn.Parameter, m.weight)
                weight.copy_(torch.randn_like(weight))
                weight.div_(weight.norm(dim=1, keepdim=True))
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, (GraphNorm, LayerNorm)):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)
