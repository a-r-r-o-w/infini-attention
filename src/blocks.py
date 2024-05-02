from typing import Optional

import torch
import torch.nn as nn

from .attention import InfiniAttention


T = torch.FloatTensor


class PositionwiseFFN(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x: T) -> T:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        query_key_dim: int,
        value_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        self.attn = InfiniAttention(embedding_dim, query_key_dim, value_dim, num_heads)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.pffn = PositionwiseFFN(embedding_dim, ffn_dim)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x: T, mask: Optional[T] = None) -> T:
        residual = x
        x = self.dropout1(self.attn(x, x, x, mask))
        x = self.norm1(residual + x)

        residual = x
        x = self.dropout2(self.pffn(x))
        x = self.norm2(residual + x)

        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        query_key_dim: int,
        value_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        self.attn1 = InfiniAttention(embedding_dim, query_key_dim, value_dim, num_heads)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.attn2 = InfiniAttention(embedding_dim, query_key_dim, value_dim, num_heads)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.pffn = PositionwiseFFN(embedding_dim, ffn_dim)
        self.dropout3 = nn.Dropout(p=dropout_rate)
        self.norm3 = nn.LayerNorm(embedding_dim)

    def forward(
        self, x: T, enc_x: T, mask: Optional[T] = None, dec_enc_mask: Optional[T] = None
    ) -> T:
        residual = x
        x = self.dropout1(self.attn1(x, x, x, mask))
        x = self.norm1(residual + x)

        residual = x
        x = self.dropout2(self.attn2(x, enc_x, enc_x, dec_enc_mask))
        x = self.norm2(residual + x)

        residual = x
        x = self.dropout3(self.pffn(x))
        x = self.norm3(residual + x)

        return x


if __name__ == "__main__":
    embedding_dim = 512
    query_key_dim = 512
    value_dim = 512
    num_heads = 8
    ffn_dim = 768
    dropout_rate = 0.1

    encoder = EncoderBlock(
        embedding_dim, query_key_dim, value_dim, num_heads, ffn_dim, dropout_rate
    )
    x = torch.randn((1, 128, embedding_dim))
    print(encoder(x).shape)
