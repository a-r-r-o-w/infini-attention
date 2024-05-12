from typing import Optional

import torch
import torch.nn as nn

from .attention import InfiniAttention


T = torch.Tensor


class PositionwiseFFN(nn.Module):
    r"""Implements the position-wise feed-forward network used in the encoder and decoder blocks.

    Args:
        embedding_dim (int):
            The dimension of the input embeddings.
        hidden_dim (int):
            The dimension of the hidden layer in the position-wise feed-forward network.
        use_pffn_bias (bool):
            Whether to use a bias in the linear layers in the position-wise feed-forward network.
    """

    def __init__(
        self, embedding_dim: int, hidden_dim: int, use_pffn_bias: bool = True
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(embedding_dim, hidden_dim, bias=use_pffn_bias)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, embedding_dim, bias=use_pffn_bias)

    def forward(self, x: T) -> T:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class EncoderBlock(nn.Module):
    r"""Implements a single encoder block in the transformer architecture.

    Args:
        embedding_dim (int):
            The dimension of the input embeddings.
        attn_head_dim (int):
            The dimension of each attention head used in the multi-head attention.
        num_query_heads (int):
            The number of query attention heads.
        num_key_value_heads (int):
            The number of key and value attention heads.
        ffn_dim (int):
            The dimension of the hidden layer in the position-wise feed-forward network.
        dropout_rate (float):
            The dropout rate used in the encoder block between sublayers.
        use_delta_update_rule (bool):
            Whether to use the delta update rule mentioned "Section 2.1.2 Compressive Memory Update"
            in the paper.
        use_attn_linear_bias (bool):
            Whether to use a bias in the linear layer after attention.
        use_pffn_bias (bool):
            Whether to use a bias in the linear layers in the position-wise feed-forward network.
    """

    def __init__(
        self,
        embedding_dim: int,
        attn_head_dim: int,
        num_query_heads: int,
        num_key_value_heads: int,
        ffn_dim: int,
        dropout_rate: float,
        use_delta_update_rule: bool = False,
        use_attn_linear_bias: bool = False,
        use_pffn_bias: bool = True,
    ) -> None:
        super().__init__()

        self.attn = InfiniAttention(
            embedding_dim,
            attn_head_dim,
            num_query_heads,
            num_key_value_heads,
            use_attn_linear_bias,
            use_delta_update_rule,
        )
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.pffn = PositionwiseFFN(embedding_dim, ffn_dim, use_pffn_bias)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x: T, mask: Optional[T] = None) -> T:
        residual = x
        x, context = self.attn(x, x, x, mask)
        x = self.norm1(residual + self.dropout1(x))

        residual = x
        x = self.dropout2(self.pffn(x))
        x = self.norm2(residual + x)

        return x, context


class DecoderBlock(nn.Module):
    r"""Implements a single decoder block in the transformer architecture.

    Args:
        embedding_dim (int):
            The dimension of the input embeddings.
        attn_head_dim (int):
            The dimension of each attention head used in the multi-head attention.
        num_query_heads (int):
            The number of query attention heads.
        num_key_value_heads (int):
            The number of key and value attention heads.
        ffn_dim (int):
            The dimension of the hidden layer in the position-wise feed-forward network.
        dropout_rate (float):
            The dropout rate used in the decoder block between sublayers.
        use_delta_update_rule (bool):
            Whether to use the delta update rule mentioned "Section 2.1.2 Compressive Memory Update"
            in the paper.
        use_attn_linear_bias (bool):
            Whether to use a bias in the linear layer after attention.
        use_pffn_bias (bool):
            Whether to use a bias in the linear layers in the position-wise feed-forward network.
    """

    def __init__(
        self,
        embedding_dim: int,
        attn_head_dim: int,
        num_query_heads: int,
        num_key_value_heads: int,
        ffn_dim: int,
        dropout_rate: float,
        use_delta_update_rule: bool = False,
        use_attn_linear_bias: bool = False,
        use_pffn_bias: bool = True,
    ) -> None:
        super().__init__()

        self.attn1 = InfiniAttention(
            embedding_dim,
            attn_head_dim,
            num_query_heads,
            num_key_value_heads,
            use_attn_linear_bias,
            use_delta_update_rule,
        )
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.attn2 = InfiniAttention(
            embedding_dim,
            attn_head_dim,
            num_query_heads,
            num_key_value_heads,
            use_attn_linear_bias,
            use_delta_update_rule,
        )
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.pffn = PositionwiseFFN(embedding_dim, ffn_dim, use_pffn_bias)
        self.dropout3 = nn.Dropout(p=dropout_rate)
        self.norm3 = nn.LayerNorm(embedding_dim)

    def forward(
        self, x: T, enc_x: T, mask: Optional[T] = None, dec_enc_mask: Optional[T] = None
    ) -> T:
        residual = x
        x, context_self_attn = self.attn1(x, x, x, mask)
        x = self.norm1(residual + self.dropout1(x))

        residual = x
        x, context_enc_dec_attn = self.attn2(x, enc_x, enc_x, dec_enc_mask)
        x = self.norm2(residual + self.dropout2(x))

        residual = x
        x = self.dropout3(self.pffn(x))
        x = self.norm3(residual + x)

        return x, (context_self_attn, context_enc_dec_attn)
