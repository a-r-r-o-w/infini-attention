import torch
import torch.nn as nn

T = torch.Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_length: int) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.max_length = max_length

        two_i = torch.arange(0, embedding_dim, 2)
        numerator = torch.arange(0, max_length).unsqueeze(dim=1)
        denominator = 10000.0 ** (two_i / embedding_dim)
        x = numerator / denominator

        pe = torch.zeros((max_length, embedding_dim))
        pe[:, 0::2] = torch.sin(x)
        pe[:, 1::2] = torch.cos(x)

        self.register_parameter("pe", nn.Parameter(pe, requires_grad=False))
        self.pe: T

    def forward(self, x: T) -> T:
        batch_size, seq_length, embedding_dim = x.shape
        return x + self.pe[:seq_length, :]
