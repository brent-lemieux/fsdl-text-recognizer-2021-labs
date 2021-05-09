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
        hidden_dims = self.args.get("hidden_dims", HIDDEN_DIMS)
        dropout = self.args.get("dropout", DROPOUT)

        self.dropout = nn.Dropout(dropout)

        # Set hidden layers.
        self.fc_hidden =[nn.Linear(input_dim, hidden_dims[0])]
        for i, n_neurons in enumerate(hidden_dims):
            if i + 1 < len(hidden_dims):
                fc = nn.Linear(n_neurons, hidden_dims[i+1])
                self.fc_hidden.append(fc)
        # Set output layer.
        self.output = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        for fc in self.fc_hidden:
            x = fc(x)
            x = F.relu(x)
            x = self.dropout(x)
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        # x = F.relu(x)
        # x = self.dropout(x)
        x = self.output(x)
        return x

    @staticmethod
    def add_to_argparse(parser):
        # parser.add_argument("--fc1", type=int, default=FC1_DIM)
        # parser.add_argument("--fc2", type=int, default=FC2_DIM)
        parser.add_argument("--hidden_dims", type=ast.literal_eval, default=HIDDEN_DIMS)
        parser.add_argument("--dropout", type=float, default=DROPOUT)
        return parser
