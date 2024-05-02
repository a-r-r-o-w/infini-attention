# infini-attention

Possibly faithful implementation of the "Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention" paper by Google Research.

- Paper: https://arxiv.org/abs/2404.07143

I've been trying to practice implementing papers related to new model architectures for the past few months, in the hope to become more confident with large codebases and new ideas.

I've recorded myself walking through the paper and implementing it in a two hour programming stream on [YouTube](https://youtu.be/SLrSJSL4pdk). For a great explanation on InfiniAttention, watch [this](https://youtu.be/r_UBBfTPcF0) video by Yannic Kilcher and [this](https://youtu.be/MRTTGMlKgb8) video by Gabriel Mongaras.

### TLDR

InfiniAttention is a method presented in the linked paper that proposes an efficient addition to multi-head attention. The method adds a compressive memory into the vanilla MHA from [Attention Is All You Need](https://arxiv.org/abs/1706.03762). This allows transformer architectures to scale to infinite context while only using of fixed amount of memory.

TODO

### Usage

TODO

### TODO

- [ ] Verify if attention mechanism for memory is actually correctly implemented
- [ ] Implement the training loop
- [ ] Implement passkey retrieval and other needle-in-a-haystack experiments
- [ ] Improve documentation
- [ ] Implement RoPE instead of current PE
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
