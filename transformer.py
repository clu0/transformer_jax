from jax import numpy as jnp
from flax import linen as nn


class MLP(nn.Module):
    n_embd: int
    n_inner: int
    p_dropout: float = 0.5
    
    def setup(self):
        self.dense1 = nn.Dense(self.n_inner)
        self.drop1 = nn.Dropout(rate=self.p_dropout)
        self.dense2 = nn.Dense(self.n_embd)
        self.drop2 = nn.Dropout(rate=self.p_dropout)
        
    def __call__(self, x, training: bool):
        x = self.dense1(x)
        x = nn.gelu(x)
        x = self.drop1(x, deterministic=not training)
        x = self.dense2(x)
        x = self.drop2(x, deterministic=not training)
        return x


class CausalMultiheadAttention(nn.Module):
    n_embd: int
    heads: int
    p_attn_dropout: float = 0.5
    p_out_dropout: float = 0.5

    def setup(self):
        assert self.n_embd % self.heads == 0
        self.dim_per_head = self.n_embd // self.heads
        
        self.qkv = nn.Dense(3 * self.n_embd)
        self.attn_dropout = nn.Dropout(rate=self.p_attn_dropout)
        self.out_dropout = nn.Dropout(rate=self.p_out_dropout)
        
    def __call__(self, x, training: bool):
        B, T, _ = x.shape
        q, k, v = jnp.split(self.qkv(x), 3, axis=-1)
        q = q.reshape(B, T, self.heads, self.dim_per_head)
        k = k.reshape(B, T, self.heads, self.dim_per_head)
        v = v.reshape(B, T, self.heads, self.dim_per_head)
        
        attn = jnp.einsum("bthd,bThd->bhtT", q, k) / jnp.sqrt(self.dim_per_head)  # (B, H, T, T)
        
        # causal mask
        causal_mask = jnp.tril(jnp.ones((T, T)))
        bias = jnp.full((T, T), -jnp.inf)
        attn = jnp.where(causal_mask == 0, bias, attn)
        attn = nn.softmax(attn, axis=-1)
        attn = self.attn_dropout(attn, deterministic=not training)
        
        out = jnp.einsum("bhtT,bThd->bthd", attn, v).reshape(B, T, self.n_embd)
        out = self.out_dropout(out, deterministic=not training)
        return out


class DecoderBlock(nn.Module):
    n_embd: int
    heads: int
    n_inner: int
    attn_dropout: float = 0.5
    mlp_dropout: float = 0.5
    out_dropout: float = 0.5
    
    def setup(self):
        self.ln1 = nn.LayerNorm()
        self.mlp = MLP(self.n_embd, self.n_inner, self.mlp_dropout)
        self.ln2 = nn.LayerNorm()
        self.attn = CausalMultiheadAttention(self.n_embd, self.heads, self.attn_dropout, self.out_dropout)
    
    def __call__(self, x, training: bool):
        r = self.ln1(x)
        r = self.attn(r, training=training)
        x = x + r
        r = self.ln2(x)
        r = self.mlp(r, training=training)
        x = x + r
        return x


class Decoder(nn.Module):
    n_layers: int
    n_vocab: int
    block_size: int
    n_embd: int
    heads: int
    n_inner: int
    attn_dropout: float = 0.5
    mlp_dropout: float = 0.5
    out_dropout: float = 0.5
    
    def setup(self):
        self.token_embd = nn.Embed(self.n_vocab, self.n_embd)
        self.timestep_embd = nn.Embed(self.block_size, self.n_embd)
        self.blocks = [DecoderBlock(self.n_embd, self.heads, self.n_inner, self.attn_dropout, self.mlp_dropout, self.out_dropout) for _ in range(self.n_layers)]
        self.final_ln = nn.LayerNorm()
        self.logits = nn.Dense(self.n_vocab)
        
    def __call__(self, x, training: bool):
        x = self.token_embd(x)
        x = x + self.timestep_embd(jnp.arange(x.shape[1]))
        for block in self.blocks:
            x = block(x, training=training)
        x = self.final_ln(x)
        x = self.logits(x)
        return x