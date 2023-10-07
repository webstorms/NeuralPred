import torch.nn as nn
import torch.nn.functional as F
from brainbox import models


class Readout(models.BBModel):

    def __init__(self, n_in, n_out):
        super().__init__()
        self._n_in = n_in
        self._n_out = n_out
        self._linear = nn.Linear(n_in, n_out)
        self.init_weight(self._linear.weight, "glorot_uniform")

    @property
    def hyperparams(self):
        return {**super().hyperparams, "n_in": self._n_in, "n_out": self._n_out}

    def get_params(self):
        return [self._linear.weight]

    def forward(self, x):
        # x: b x n
        x = self._linear(x)
        x = F.softplus(x, 1, 20)

        return x

