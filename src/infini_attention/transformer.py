import torch
import torch.nn as nn

from .positional_encoding import PositionalEncoding
from .blocks import DecoderBlock, EncoderBlock


T = torch.Tensor


class EncoderDecoderTransformer(nn.Module):
    r"""Implements the encoder-decoder transformer architecture.

    Args:
        num_enc_layers (int):
            The number of encoder layers in the transformer.
        num_dec_layers (int):
            The number of decoder layers in the transformer.
        src_vocab_size (int):
            The size of the source vocabulary.
        tgt_vocab_size (int):
            The size of the target vocabulary.
        src_pad_idx (int):
            The index of the padding token in the source vocabulary.
        tgt_pad_idx (int):
            The index of the padding token in the target vocabulary.
        max_length (int):
            The maximum length of the sequence.
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
            The dropout rate used in the encoder and decoder blocks between sublayers.
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
        num_enc_layers: int,
        num_dec_layers: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_idx: int,
        tgt_pad_idx: int,
        max_length: int,
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

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        self.src_embedding = nn.Embedding(src_vocab_size, embedding_dim)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embedding_dim)
        self.pe = PositionalEncoding(embedding_dim, max_length)
        self.encoders = nn.ModuleList(
            [
                EncoderBlock(
                    embedding_dim,
                    attn_head_dim,
                    num_query_heads,
                    num_key_value_heads,
                    ffn_dim,
                    dropout_rate,
                    use_delta_update_rule,
                    use_attn_linear_bias,
                    use_pffn_bias,
                )
                for _ in range(num_enc_layers)
            ]
        )
        self.decoders = nn.ModuleList(
            [
                DecoderBlock(
                    embedding_dim,
                    attn_head_dim,
                    num_query_heads,
                    num_key_value_heads,
                    ffn_dim,
                    dropout_rate,
                    use_delta_update_rule,
                    use_attn_linear_bias,
                    use_pffn_bias,
                )
                for _ in range(num_dec_layers)
            ]
        )

        self.scale = embedding_dim**0.5

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
        causal_mask = torch.tril(
            torch.ones((1, seq_length, seq_length), device=x.device)
        ).bool()
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
        encoder_contexts = []
        for block in self.encoders:
            src, encoder_context = block(src, attn_mask)
            encoder_contexts.append(encoder_context)

        # 4. Decoder
        decoder_contexts = []
        for block in self.decoders:
            tgt, decoder_context = block(tgt, src, causal_mask, attn_mask)
            decoder_contexts.append(decoder_context)

        return tgt, encoder_contexts, decoder_contexts

    def clear_memory(self) -> None:
        r"""Clears the InfiniAttention memory of the encoder and decoder blocks."""

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
