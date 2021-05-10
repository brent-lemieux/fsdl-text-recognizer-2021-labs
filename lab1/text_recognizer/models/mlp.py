import argparse
import ast
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# FC1_DIM = 1024
# FC2_DIM = 128
HIDDEN_DIMS = [256, 128]
DROPOUT = 0.5


class MLP(nn.Module):
    """Simple MLP suitable for recognizing single characters."""

    def __init__(
        self,
        data_config: Dict[str, Any],
        args: argparse.Namespace = None,
    ) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}

        input_dim = np.prod(data_config["input_dims"])
        num_classes = len(data_config["mapping"])

        # fc1_dim = self.args.get("fc1", FC1_DIM)
        # fc2_dim = self.args.get("fc2", FC2_DIM)
        dropout = self.args.get("dropout", DROPOUT)

        self.dropout = nn.Dropout(dropout)

        # Set hidden layers.
        hidden_dims = self.args.get("hidden_dims", HIDDEN_DIMS)
        layer_dims = [input_dim] + hidden_dims + [num_classes]

        self.layers = nn.ModuleList([
            nn.Linear(dim, layer_dims[i+1])
            for i, dim in enumerate(layer_dims[:-1])
        ])

    def forward(self, x):
        x = torch.flatten(x, 1)
        for i, fc in enumerate(self.layers[:-1]):
            x = fc(x)
            x = F.relu(x)
            x = self.dropout(x)
        output_layer = self.layers[-1]
        x = output_layer(x)
        return x

    @staticmethod
    def add_to_argparse(parser):
        # parser.add_argument("--fc1", type=int, default=FC1_DIM)
        # parser.add_argument("--fc2", type=int, default=FC2_DIM)
        parser.add_argument("--hidden_dims", type=ast.literal_eval, default=HIDDEN_DIMS)
        parser.add_argument("--dropout", type=float, default=DROPOUT)
        return parser
