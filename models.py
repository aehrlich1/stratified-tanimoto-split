import sys

import torch
import torch_geometric.utils.smiles as pyg_smiles
from torch import nn
from torch_geometric.nn import GIN
from torch_geometric.nn.pool import global_add_pool


class GINModel(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        encoding_dim: int,
    ):
        super().__init__()
        self.encoding_model = CategoricalEncodingModel(embedding_dim=encoding_dim)
        in_channels = self.encoding_model.get_feature_embedding_dim()

        self.model = GIN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.pool = global_add_pool

        self.projection = nn.Sequential(
            nn.Linear(out_channels, out_channels * 2),
            nn.ReLU(),
            nn.Linear(out_channels * 2, out_channels * 2),
            nn.ReLU(),
            nn.Linear(out_channels * 2, 1),
        )

    def forward(self, data):
        data = self.encoding_model(data)
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        h = self.model(x=x, edge_index=edge_index, edge_attr=edge_attr)
        h_G = self.pool(x=h, batch=data.batch)
        out = self.projection(h_G)

        return out

    def reset_parameters(self):
        self.model.reset_parameters()
        for module in self.projection.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()


class CategoricalEncodingModel(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.node_embedding = CategoricalEmbeddingModel(
            category_type="node", embedding_dim=embedding_dim
        )
        self.edge_embedding = CategoricalEmbeddingModel(
            category_type="edge", embedding_dim=embedding_dim
        )

    def forward(self, data):
        data.x = self.node_embedding(data.x)
        data.edge_attr = self.edge_embedding(data.edge_attr)

        return data

    def get_feature_embedding_dim(self):
        return self.node_embedding.get_node_feature_dim()

    def get_edge_embedding_dim(self):
        return self.edge_embedding.get_edge_feature_dim()


class CategoricalEmbeddingModel(nn.Module):
    """
    Model to embed categorical node or edge features
    """

    def __init__(self, category_type, embedding_dim=8):
        super().__init__()
        if category_type == "node":
            num_categories = self._get_num_node_categories()
        elif category_type == "edge":
            num_categories = self._get_num_edge_categories()
        else:
            print("Invalid category type")
            sys.exit()
        self.embedding_dim = embedding_dim
        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_categories[i], embedding_dim) for i in range(len(num_categories))]
        )

    def forward(self, x):
        embedded_vars = [self.embeddings[i](x[:, i]) for i in range(len(self.embeddings))]

        return torch.cat(embedded_vars, dim=-1)

    def get_node_feature_dim(self):
        return len(self._get_num_node_categories() * self.embedding_dim)

    def get_edge_feature_dim(self):
        return len(self._get_num_edge_categories() * self.embedding_dim)

    @staticmethod
    def _get_num_node_categories() -> list[int]:
        return [
            len(pyg_smiles.x_map[prop]) for prop in pyg_smiles.x_map
        ]  # [119, 9, 11, 12, 9, 5, 8, 2, 2]

    @staticmethod
    def _get_num_edge_categories() -> list[int]:
        return [len(pyg_smiles.e_map[prop]) for prop in pyg_smiles.e_map]  # [22, 6, 2]
