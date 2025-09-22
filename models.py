"""
This file will contain all our models
"""

from torch import nn
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

    def forward(self, data):
        data = self.encoding_model(data)
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        h = self.model(x=x, edge_index=edge_index, edge_attr=edge_attr)
        h_G = self.pool(x=h, batch=data.batch)

        return h_G
