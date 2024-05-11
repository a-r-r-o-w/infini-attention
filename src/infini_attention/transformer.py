import torch
import torch.nn as nn

from .positional_encoding import PositionalEncoding
from .blocks import DecoderBlock, EncoderBlock


T = torch.Tensor


class EncoderDecoderTransformer(nn.Module):
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
        query_key_dim: int,
        value_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout_rate: float,
        use_query_bias: bool = False,
        use_key_bias: bool = False,
        use_value_bias: bool = False,
        use_output_bias: bool = False,
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
                    query_key_dim,
                    value_dim,
                    num_heads,
                    ffn_dim,
                    dropout_rate,
                    use_query_bias,
                    use_key_bias,
                    use_value_bias,
                    use_output_bias,
                    use_pffn_bias,
                )
                for _ in range(num_enc_layers)
            ]
        )
        self.decoders = nn.ModuleList(
            [
                DecoderBlock(
                    embedding_dim,
                    query_key_dim,
                    value_dim,
                    num_heads,
                    ffn_dim,
                    dropout_rate,
                    use_query_bias,
                    use_key_bias,
                    use_value_bias,
                    use_output_bias,
                    use_pffn_bias,
                )
                for _ in range(num_dec_layers)
            ]
        )

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
