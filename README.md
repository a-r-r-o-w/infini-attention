# infini-attention

Possibly faithful implementation of the "Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention" paper by Google Research.

- Paper: https://arxiv.org/abs/2404.07143

I've recorded myself walking through the paper and implementing it in a two hour programming stream on [YouTube](https://youtu.be/SLrSJSL4pdk). For a great explanation on InfiniAttention, watch [this](https://youtu.be/r_UBBfTPcF0) video by Yannic Kilcher and [this](https://youtu.be/MRTTGMlKgb8) video by Gabriel Mongaras.

### TLDR

InfiniAttention is a method presented in the linked paper that proposes an efficient addition to multi-head attention. The method adds a compressive memory into the vanilla MHA from [Attention Is All You Need](https://arxiv.org/abs/1706.03762). This allows transformer architectures to scale to infinite context while only using of fixed amount of memory.

TODO

### Setup

```bash
git clone https://github.com/a-r-r-o-w/infini-attention
cd infini-attention

pip install -r requirements.txt
pip install -e .
python3 setup.py develop
```

### Usage

```python
import torch
from infini_attention import EncoderDecoderTransformer

model = EncoderDecoderTransformer(
    num_enc_layers=3,
    num_dec_layers=3,
    src_vocab_size=5000,
    tgt_vocab_size=5000,
    src_pad_idx=1,
    tgt_pad_idx=1,
    max_length=10000,
    embedding_dim=512,
    query_key_dim=512,
    value_dim=512,
    num_heads=8,
    ffn_dim=2048,
    dropout_rate=0.1,
    use_pffn_bias=True,
)

batch_size = 32
seq_length = 1024
src_ids = torch.randint(0, 5000, (batch_size, seq_length))
tgt_ids = torch.randint(0, 5000, (batch_size, seq_length))

print(model(src_ids, tgt_ids).shape) # (batch_size, seq_length, embedding_dim)
```

### TODO

- [ ] Verify if attention mechanism for memory is actually correctly implemented
- [ ] Implement training loop and sequence segmentation
- [ ] Implement passkey retrieval and other needle-in-a-haystack experiments
- [ ] Improve documentation
- [ ] Implement RoPE
- [ ] Implement SwiGLU and other activation variants
- [ ] Implement GQA
- [ ] Add encoder/decoder-only variants

### Citations

```
@misc{munkhdalai2024leave,
      title={Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention}, 
      author={Tsendsuren Munkhdalai and Manaal Faruqui and Siddharth Gopal},
      year={2024},
      eprint={2404.07143},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

```
@misc{vaswani2023attention,
    title={Attention Is All You Need}, 
    author={Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
    year={2023},
    eprint={1706.03762},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
