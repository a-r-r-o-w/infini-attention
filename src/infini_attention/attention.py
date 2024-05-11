from typing import Optional

import torch
import torch.nn as nn


T = torch.Tensor


class ScaledDotProductAttention(nn.Module):
    def __init__(self, query_key_dim: int) -> None:
        super().__init__()

        self.query_key_dim = query_key_dim
        self.scale = torch.sqrt(torch.tensor(query_key_dim))

        self.softmax = nn.Softmax(dim=3)

    def forward(self, query: T, key: T, value: T, mask: Optional[T] = None) -> T:
        # query: [batch_size, num_heads, seq_length, query_key_dim]
        #   key: [batch_size, num_heads, seq_length, query_key_dim]
        # value: [batch_size, num_heads, seq_length, value_dim]

        # 1. Matmul
        key_T = key.transpose(2, 3)
        x = torch.matmul(
            query, key_T
        )  # [batch_size, num_heads, seq_length, seq_length]

        # 2. Scale
        x /= self.scale

        # 3. Mask
        if mask is not None:
            x = x.masked_fill(mask == False, value=-1e9)

        # 4. Softmax
        context = self.softmax(x)

        # 5. Matmul
        x = torch.matmul(
            context, value
        )  # [batch_size, num_heads, seq_length, value_dim]

        return x, context


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        attn_head_dim: int,
        num_query_heads: int,
        num_key_value_heads: int,
        use_attn_linear_bias: bool = False,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.attn_head_dim = attn_head_dim
        self.query_dim = attn_head_dim * num_query_heads
        self.key_value_dim = attn_head_dim * num_key_value_heads
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads

        self.q_proj = nn.Linear(embedding_dim, self.query_dim, bias=False)
        self.k_proj = nn.Linear(embedding_dim, self.key_value_dim, bias=False)
        self.v_proj = nn.Linear(embedding_dim, self.key_value_dim, bias=False)
        self.attn = ScaledDotProductAttention(attn_head_dim)
        self.linear = nn.Linear(
            self.key_value_dim, embedding_dim, bias=use_attn_linear_bias
        )

    def forward(self, query: T, key: T, value: T, mask: Optional[T] = None) -> T:
        # 1. Projection
        q_proj: T = self.q_proj(query)
        k_proj: T = self.k_proj(key)
        v_proj: T = self.v_proj(value)

        # 2. Split into attention heads and transpose to get expected shape
        batch_size, seq_length, _ = q_proj.shape
        q_proj = q_proj.view(batch_size, -1, self.num_query_heads, self.attn_head_dim)
        k_proj = k_proj.view(
            batch_size, -1, self.num_key_value_heads, self.attn_head_dim
        )
        v_proj = v_proj.view(
            batch_size, -1, self.num_key_value_heads, self.attn_head_dim
        )

        q_proj = q_proj.transpose(1, 2)
        k_proj = k_proj.transpose(1, 2)
        v_proj = v_proj.transpose(1, 2)

        # 3. SDPA
        x, context = self.attn(q_proj, k_proj, v_proj, mask)

        # 4. Concat
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.key_value_dim)

        # 5. Linear
        x = self.linear(x)

        return x, context


class InfiniAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        attn_head_dim: int,
        num_query_heads: int,
        num_key_value_heads: int,
        use_attn_linear_bias: bool = False,
        use_delta_update_rule: bool = False,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.attn_head_dim = attn_head_dim
        self.query_dim = attn_head_dim * num_query_heads
        self.key_value_dim = attn_head_dim * num_key_value_heads
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.use_delta_update_rule = use_delta_update_rule

        self.q_proj = nn.Linear(embedding_dim, self.query_dim, bias=False)
        self.k_proj = nn.Linear(embedding_dim, self.key_value_dim, bias=False)
        self.v_proj = nn.Linear(embedding_dim, self.key_value_dim, bias=False)
        self.attn = ScaledDotProductAttention(attn_head_dim)
        self.linear = nn.Linear(
            self.key_value_dim, embedding_dim, bias=use_attn_linear_bias
        )
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

        # beta is used in long-term context injection for each attention head
        # [b? n 1 1] x [b n s v] => [b n s v]
        self.beta = nn.Parameter(torch.randn((num_key_value_heads, 1, 1)))

        #    key: [batch_size, num_key_value_heads, seq_length, attn_head_dim]
        #  key_T: [batch_size, num_key_value_heads, attn_head_dim, seq_length]
        # memory: dim(key_T) * dim(value)
        #         [batch_size, num_kv_heads, attn_head_dim, seq_length] * [batch_size, num_kv_heads, seq_length, attn_head_dim]
        #      => [batch_size, num_kv_heads, attn_head_dim, attn_head_dim]
        memory = torch.zeros((num_key_value_heads, attn_head_dim, attn_head_dim))
        z = torch.zeros((num_key_value_heads, self.attn_head_dim))
        self.register_buffer("memory", memory)
        self.register_buffer("z", z)

    def forward(self, query: T, key: T, value: T, mask: Optional[T] = None) -> T:
        # 1. Projection
        q_proj: T = self.q_proj(query)
        k_proj: T = self.k_proj(key)
        v_proj: T = self.v_proj(value)

        # 2. Split into attention heads and transpose to get expected shape
        batch_size, seq_length, _ = q_proj.shape
        q_proj = q_proj.view(batch_size, -1, self.num_query_heads, self.attn_head_dim)
        k_proj = k_proj.view(
            batch_size, -1, self.num_key_value_heads, self.attn_head_dim
        )
        v_proj = v_proj.view(
            batch_size, -1, self.num_key_value_heads, self.attn_head_dim
        )

        q_proj = q_proj.transpose(1, 2)
        k_proj = k_proj.transpose(1, 2)
        v_proj = v_proj.transpose(1, 2)

        # 2.1 Retrieve from memory
        elu_q = self.elu(q_proj) + 1

        # numerator: [b n_q s a] x [b? n_kv a a] => [b n??? s a]
        # denominotor: [b n_q s e] x [b? n_kv e 1] => [b n??? s 1]
        # TODO: n??? For now, num_query_heads must be equal to num_key_value_heads otherwise this will fail
        A_mem = torch.matmul(elu_q, self.memory) / torch.matmul(
            elu_q, self.z.unsqueeze(dim=-1)
        )

        # 2.2 Memory update
        elu_k: T = self.elu(k_proj) + 1
        elu_k_T = elu_k.transpose(2, 3)

        if self.use_delta_update_rule:
            v_delta = torch.matmul(elu_k, self.memory) / torch.matmul(
                elu_k, self.z.unsqueeze(dim=-1)
            )
            v = v_proj - v_delta
        else:
            v = v_proj

        self.memory = self.memory + torch.matmul(elu_k_T, v)
        self.z = self.z + elu_k.sum(dim=2)

        # 3. SDPA
        A_dot, context = self.attn(q_proj, k_proj, v_proj, mask)

        # 3.1 Long-term context injection
        beta = self.sigmoid(self.beta)
        x = beta * A_mem + (1 - beta) * A_dot

        # 4. Concat
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.key_value_dim)

        # 5. Linear
        x = self.linear(x)

        return x, context
