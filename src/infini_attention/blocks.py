from typing import Optional

import torch
import torch.nn as nn

from .attention import InfiniAttention


T = torch.Tensor


class PositionwiseFFN(nn.Module):
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
