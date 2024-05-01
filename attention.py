from typing import Optional

import torch
import torch.nn as nn


T = torch.FloatTensor

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
        x = torch.matmul(query, key_T) # [batch_size, num_heads, seq_length, seq_length]

        # 2. Scale
        x /= self.scale

        # 3. Mask
        if mask is not None:
            x = x.masked_fill(mask == False, value=1e-9)
        
        # 4. Softmax
        x = self.softmax(x)

        # 5. Matmul
        x = torch.matmul(x, value) # [batch_size, num_heads, seq_length, value_dim]

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, query_key_dim: int, value_dim: int, num_heads: int) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.query_key_dim = query_key_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.query_key_dim_per_head = query_key_dim // num_heads
        self.value_dim_per_head = value_dim // num_heads

        assert self.query_key_dim_per_head * num_heads == query_key_dim
        assert self.value_dim_per_head * num_heads == value_dim

        self.q_proj = nn.Linear(embedding_dim, query_key_dim)
        self.k_proj = nn.Linear(embedding_dim, query_key_dim)
        self.v_proj = nn.Linear(embedding_dim, value_dim)
        self.attn = ScaledDotProductAttention(query_key_dim)
        self.linear = nn.Linear(value_dim, embedding_dim)
    
    def forward(self, query: T, key: T, value: T, mask: Optional[T] = None) -> T:
        # 1. Projection
        q_proj: T = self.q_proj(query)
        k_proj: T = self.k_proj(key)
        v_proj: T = self.v_proj(value)

        # 2. Split into attention heads and transpose to get expected shape
        batch_size, seq_length, _ = q_proj.shape
        q_proj = q_proj.view(batch_size, seq_length, self.num_heads, self.query_key_dim_per_head)
        k_proj = k_proj.view(batch_size, seq_length, self.num_heads, self.query_key_dim_per_head)
        v_proj = v_proj.view(batch_size, seq_length, self.num_heads, self.value_dim_per_head)

        q_proj = q_proj.transpose(1, 2)
        k_proj = k_proj.transpose(1, 2)
        v_proj = v_proj.transpose(1, 2)

        # 3. SDPA
        x: T = self.attn(q_proj, k_proj, v_proj, mask)

        # 4. Concat
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, seq_length, self.value_dim)

        # 5. Linear
        x = self.linear(x)

        return x


class InfiniAttention(nn.Module):
    def __init__(self, embedding_dim: int, query_key_dim: int, value_dim: int, num_heads: int) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.query_key_dim = query_key_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.query_key_dim_per_head = query_key_dim // num_heads
        self.value_dim_per_head = value_dim // num_heads

        assert self.query_key_dim_per_head * num_heads == query_key_dim
        assert self.value_dim_per_head * num_heads == value_dim

        self.q_proj = nn.Linear(embedding_dim, query_key_dim)
        self.k_proj = nn.Linear(embedding_dim, query_key_dim)
        self.v_proj = nn.Linear(embedding_dim, value_dim)
        self.attn = ScaledDotProductAttention(query_key_dim)
        self.linear = nn.Linear(value_dim, embedding_dim)
        self.beta = nn.Parameter(torch.randn(1))
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

        # key: [batch_size, num_heads, seq_length, query_key_dim_per_head]
        # key_T: [batch_size, num_heads, query_key_dim_per_head, seq_length]
        # memory: dim(key_T) * dim(value) [batch_size, num_heads, query_key_dim_per_head, seq_length] * [batch_size, num_heads, seq_length, value_dim_per_head]
        #         => [batch_size, num_heads, query_key_dim_per_head, value_dim_per_head]
        self.memory = torch.zeros((num_heads, self.query_key_dim_per_head, self.value_dim_per_head))
        self.z = torch.zeros((num_heads, self.query_key_dim_per_head))
    
    def forward(self, query: T, key: T, value: T, mask: Optional[T] = None) -> T:
        # 1. Projection
        q_proj: T = self.q_proj(query)
        k_proj: T = self.k_proj(key)
        v_proj: T = self.v_proj(value)

        # 2. Split into attention heads and transpose to get expected shape
        batch_size, seq_length, _ = q_proj.shape
        q_proj = q_proj.view(batch_size, seq_length, self.num_heads, self.query_key_dim_per_head)
        k_proj = k_proj.view(batch_size, seq_length, self.num_heads, self.query_key_dim_per_head)
        v_proj = v_proj.view(batch_size, seq_length, self.num_heads, self.value_dim_per_head)

        q_proj = q_proj.transpose(1, 2)
        k_proj = k_proj.transpose(1, 2)
        v_proj = v_proj.transpose(1, 2)

        # 2.1 Retrieve from memory
        elu_q = self.elu(q_proj) + 1
        
        # numerator: [b n s e] x [b? n e v] => [b n s v]
        # denominotor: [b n s e] x [b? n e 1] => [b n s 1]
        A_mem = torch.matmul(elu_q, self.memory) / torch.matmul(elu_q, self.z.unsqueeze(dim=-1))

        # 2.2 Memory update
        elu_k: T = self.elu(k_proj) + 1
        elu_k_T = elu_k.transpose(2, 3)

        # [n e v] += [n e v] + (b n e s) * (b n s v)
        # [n e v] += [n e v] + [b n e v]
        self.memory = self.memory + torch.matmul(elu_k_T, v_proj)
        self.z = self.z + elu_k.sum(dim=2)
        # TODO: Implement delta rule

        # 3. SDPA
        A_dot: T = self.attn(q_proj, k_proj, v_proj, mask)

        # 3.1 Long-term context injection
        beta = self.sigmoid(self.beta)
        x = beta * A_mem + (1 - beta) * A_dot

        # 4. Concat
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, seq_length, self.value_dim)

        # 5. Linear
        x = self.linear(x)

        return x


if __name__ == "__main__":
    embedding_dim = 512
    query_key_dim = 512
    value_dim = 512
    num_heads = 8

    # mha = MultiHeadAttention(embedding_dim, query_key_dim, value_dim, num_heads)
    mha = InfiniAttention(embedding_dim, query_key_dim, value_dim, num_heads)
    x = torch.randn((1, 128, embedding_dim))

    print(mha(x, x, x).shape)
