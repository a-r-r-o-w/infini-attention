from typing import Optional

import torch
import torch.nn as nn

from attention import InfiniAttention


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


class EncoderBlock(nn.Module):
    def __init__(self, embedding_dim: int, query_key_dim: int, value_dim: int, num_heads: int, ffn_dim: int, dropout_rate: float) -> None:
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
    def __init__(self, embedding_dim: int, query_key_dim: int, value_dim: int, num_heads: int, ffn_dim: int, dropout_rate: float) -> None:
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
    
    def forward(self, x: T, enc_x: T, mask: Optional[T] = None, dec_enc_mask: Optional[T] = None) -> T:
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


class EncoderDecoderTransformer(nn.Module):
    def __init__(self, num_enc_layers: int, num_dec_layers: int, src_vocab_size: int, tgt_vocab_size: int, src_pad_idx: int, tgt_pad_idx: int, max_length: int, embedding_dim: int, query_key_dim: int, value_dim: int, num_heads: int, ffn_dim: int, dropout_rate: float) -> None:
        super().__init__()

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        
        self.src_embedding = nn.Embedding(src_vocab_size, embedding_dim)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embedding_dim)
        self.pe = PositionalEncoding(embedding_dim, max_length)
        self.encoders = nn.ModuleList([
            EncoderBlock(embedding_dim, query_key_dim, value_dim, num_heads, ffn_dim, dropout_rate)
            for _ in range(num_enc_layers)
        ])
        self.decoders = nn.ModuleList([
            DecoderBlock(embedding_dim, query_key_dim, value_dim, num_heads, ffn_dim, dropout_rate)
            for _ in range(num_dec_layers)
        ])

        self.scale = torch.sqrt(torch.tensor(embedding_dim))
    
    def _get_attn_mask(self, x: T, pad_idx: int) -> torch.BoolTensor:
        # [batch_size, seq_length] -> [batch_size, 1, 1, seq_length]
        return (x != pad_idx).bool().unsqueeze(1).unsqueeze(2)
    
    def _get_causal_mask(self, x: T, pad_idx: int) -> torch.BoolTensor:
        # [batch_size, seq_length] -> [batch_size, 1, 1, seq_length]
        # [1 0 0 0]
        # [1 1 0 0]
        # [1 1 1 0]
        # [1 1 1 1]
        batch_size, seq_length = x.shape
        attn_mask = self._get_attn_mask(x, pad_idx)
        causal_mask = torch.tril(torch.ones((1, seq_length, seq_length))).bool()
        mask = attn_mask & causal_mask
        return mask
    
    def forward(self, src_ids: T, tgt_ids: T) -> T:
        # 0. Prepare masks
        attn_mask = self._get_attn_mask(src_ids, self.src_pad_idx)
        causal_mask = self._get_causal_mask(tgt_ids, self.tgt_pad_idx)

        # 1. Get embeddings from tokens
        src = self.src_embedding(src_ids) * self.scale
        tgt = self.tgt_embedding(tgt_ids) * self.scale

        # 2. Positional encoding
        src = self.pe(src)
        tgt = self.pe(tgt)

        # 3. Encoder
        for block in self.encoders:
            src = block(src, attn_mask)

        # 4. Decoder
        for block in self.decoders:
            tgt = block(tgt, src, causal_mask, attn_mask)

        return tgt

    def clear_memory(self) -> None:
        for block in self.encoders:
            block: EncoderBlock
            block.attn.memory.zero_()
            block.attn.z.zero_()
        
        for block in self.decoders:
            block: DecoderBlock
            block.attn1.memory.zero_()
            block.attn1.z.zero_()    
            block.attn2.memory.zero_()
            block.attn2.z.zero_()


if __name__ == "__main__":
    # embedding_dim = 512
    # query_key_dim = 512
    # value_dim = 512
    # num_heads = 8
    # ffn_dim = 768
    # dropout_rate = 0.1

    # encoder = EncoderBlock(embedding_dim, query_key_dim, value_dim, num_heads, ffn_dim, dropout_rate)
    # x = torch.randn((1, 128, embedding_dim))
    # print(encoder(x).shape)

    transformer = EncoderDecoderTransformer(1, 1, 1000, 1000, 1, 1, 256, 64, 64, 64, 4, 128, 0.1)
    shape = (2, 128)
    input_ids = torch.randint(0, 1000, shape)
    tgt_ids = torch.randint(0, 1000, shape)

    print(transformer(input_ids, tgt_ids).shape)

    # for inputs, (targets_shifted or labels) in train dataloader:
    #   optimizer zero grad
    #   inputs_segmented, targets_segment = segment(inputs, targets, SEGMENT_LENGTH)
    #   for (each segment in inputs_segment, target_segmented)
    #       # run infiniattention
    #       calculate loss per segment and backward
    #   gradient clipping between 0 and 1
    #   optimizer step
    #   clear memory of infiniattention
