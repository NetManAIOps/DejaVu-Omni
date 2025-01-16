import torch as th
from einops import rearrange
from typing import List

from utils.sequential_model_builder import SequentialModelBuilder
from utils.tcn import TemporalConvNet


class TCNFeatureModule(th.nn.Module):
    def __init__(self, input_size: th.Size, embedding_size: int):
        super().__init__()
        self.x_dim = input_size[-2]
        self.n_ts = input_size[-1]
        self.n_instances = input_size[-3]
        num_channels = [embedding_size, embedding_size]
        self.z_dim = embedding_size

        self.encoder = TemporalConvNet(
            num_inputs=self.x_dim, num_channels=num_channels, kernel_size=5
        )

        self.unify_mapper = SequentialModelBuilder(
            (-1, self.n_instances, self.z_dim, self.n_ts), debug=False,
        ).add_reshape(
            -1, self.z_dim, self.n_ts,
        ).add_flatten(start_dim=-2).add_linear(embedding_size).add_reshape(
            -1, self.n_instances, embedding_size,
        ).build()

    def forward(self, x):
        z = self.encode(x)
        embedding = th.cat([self.unify_mapper(z)], dim=-1)
        return embedding

    def encode(self, input_x: th.Tensor) -> th.Tensor:
        """
        :param input_x: (batch_size, n_nodes, n_metrics, n_ts)
        :return: (batch_size, n_nodes, self.z_dim, n_ts)
        """
        batch_size, n_nodes, _, n_ts = input_x.size()
        x = rearrange(input_x, "b n m t -> (b n) m t", b=batch_size, n=n_nodes, m=self.x_dim, t=n_ts)
        assert x.size() == (batch_size * n_nodes, self.x_dim, n_ts)
        z = self.encoder(x)
        return rearrange(z, "(b n) z t -> b n z t", b=batch_size, n=n_nodes, z=self.z_dim, t=n_ts)
