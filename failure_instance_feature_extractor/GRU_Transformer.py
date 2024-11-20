import torch as th
from einops import rearrange

from utils.sequential_model_builder import SequentialModelBuilder


class GRUTransformerFeatureModule(th.nn.Module):
    def __init__(self, input_size: th.Size, embedding_size: int, num_layers: int = 3, num_heads: int = 1, num_transformer_layers: int = 3):
        super().__init__()
        self.x_dim = input_size[-2]
        self.n_ts = input_size[-1]
        self.n_instances = input_size[-3]
        self.z_dim = embedding_size
        self.num_layers = num_layers

    #     self.transformer = SequentialModelBuilder(
    #         (-1, self.x_dim, self.n_instances), debug=False
    #     ).add_transformer_encoder(
    #         num_layers=num_transformer_layers,
    #         num_heads=num_heads,
    #     ).build()
        
    #     self.metric_encoder = SequentialModelBuilder(
    #         (-1, self.x_dim), debug=False
    #     ).add_linear(
    #         self.z_dim
    #     ).build()

    #     self.gru = th.nn.GRU(
    #         input_size=self.z_dim, hidden_size=self.z_dim, num_layers=num_layers,
    #     )

    # def forward(self, x):
    #     z = self.encode(x)
    #     embedding = self.extract_temporal_features(z)
    #     return embedding

    # def encode(self, input_x: th.Tensor) -> th.Tensor:
    #     """
    #     独立地对每个时间步应用 Transformer，将 (batch_size, n_nodes, x_metrics, n_ts) 编码为 (batch_size, n_nodes, z_dim, n_ts)
    #     :param input_x: (batch_size, n_nodes, x_metrics, n_ts)
    #     :return: (batch_size, n_nodes, z_dim, n_ts)
    #     """
    #     batch_size, n_nodes, x_dim, n_ts = input_x.size()
    #     x = rearrange(input_x, "b n m t -> (b t) m n")
    #     z = self.transformer(x)
    #     z = rearrange(z, "(b t) m n -> (b t n) m", b=batch_size, n=n_nodes, m=x_dim, t=n_ts)
    #     z_m = self.metric_encoder(z)
    #     return rearrange(z_m, "(b t n) z -> b n z t", b=batch_size, n=n_nodes, z=self.z_dim, t=n_ts)

    # def extract_temporal_features(self, z: th.Tensor) -> th.Tensor:
    #     """
    #     使用 GRU 编码时间维度，将 (batch_size, n_nodes, z_dim, n_ts) 编码为 (batch_size, n_nodes, z_dim)
    #     :param z: (batch_size, n_nodes, z_dim, n_ts)
    #     :return: (batch_size, n_nodes, z_dim)
    #     """
    #     batch_size, n_nodes, z_dim, n_ts = z.size()
    #     z = rearrange(z, "b n z t -> t (b n) z", b=batch_size, n=n_nodes, z=z_dim, t=n_ts)
    #     _, h_n = self.gru(z)  # 获取最后时间步的隐状态 h_n
    #     h_n = h_n[-1]  # (batch_size * n_nodes, z_dim)
    #     return rearrange(h_n, "(b n) z -> b n z", b=batch_size, n=n_nodes, z=z_dim)


        self.encoder = th.nn.GRU(
            input_size=self.x_dim, hidden_size=self.z_dim, num_layers=num_layers,
        )

        self.unify_mapper = SequentialModelBuilder(
            (-1, self.n_instances, self.z_dim), debug=False,
        ).add_reshape(
            -1, self.z_dim, self.n_instances
        ).add_transformer_encoder(
            num_layers=num_transformer_layers,
            num_heads=num_heads,
        ).add_reshape(
            -1, self.n_instances, self.z_dim
        ).add_linear(
            embedding_size
        ).add_reshape(
            -1, self.n_instances, embedding_size,
        ).build()

    def forward(self, x):
        z = self.encode(x)
        embedding = th.cat([self.unify_mapper(z)], dim=-1)
        return embedding

    def encode(self, input_x: th.Tensor) -> th.Tensor:
        """
        :param input_x: (batch_size, n_nodes, n_metrics, n_ts)
        :return: (batch_size, n_nodes, n_ts, self.z_dim)
        """
        batch_size, n_nodes, n_metrics, n_ts = input_x.size()
        x = rearrange(input_x, "b n m t -> t (b n) m", b=batch_size, n=n_nodes, m=n_metrics, t=n_ts)
        assert x.size() == (n_ts, batch_size * n_nodes, n_metrics)
        _, h_n = self.encoder(x)
        z = h_n[-1]  # (batch_size * n_nodes, z_dim)
        return rearrange(z, "(b n) z -> b n z", b=batch_size, n=n_nodes, z=self.z_dim)
