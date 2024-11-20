import torch as th

from utils.sequential_model_builder import SequentialModelBuilder

class TransformerFeatureModule(th.nn.Module):
    def __init__(self, input_size: th.Size, embedding_size: int, num_heads: int = 1, num_layers: int = 3, has_dropout: bool = False):
        """
        :param input_size:  (batch_size, seq_length, feature_dim)
        :param embedding_size: Size of the output embedding
        :param num_heads: Number of attention heads in multi-head attention
        :param num_layers: Number of transformer layers to stack
        :param has_dropout: Whether to include dropout layers
        """
        super().__init__()
        self.input_size = input_size
        print(f"[DEBUG] {input_size=}")
        builder = SequentialModelBuilder(input_shape=input_size)

        # Optionally add dropout at the beginning
        if has_dropout:
            builder.add_dropout(0.1)

        builder.add_transformer_encoder(
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=0.1 if has_dropout else 0.0
        )

        # Reshape to combine sequence length and feature dimension
        builder.add_reshape(-1, input_size[1] * builder._output_shape[-1][1] * builder._output_shape[-1][2])  # Flattening seq_length and feature_dim
        
        # Optionally add dropout before the final linear layer
        if has_dropout:
            builder.add_dropout(0.5)

        # Linear layer to produce the output embedding
        builder.add_linear(out_features=embedding_size)

        self.builder = builder
        self.module = builder.build()

    def forward(self, input_x):
        return self.module(input_x)
