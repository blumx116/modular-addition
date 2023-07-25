from typing import NamedTuple, Optional, Tuple

import haiku as hk
import jax
import jax.nn as nn
import jax.numpy as jnp
import numpy as np
import optax
from haiku.initializers import Initializer
from jax import Array
from jax.random import KeyArray, PRNGKey
from optax import GradientTransformation, OptState, Params


class Embeddings(hk.Module):
    def __init__(
        self,
        d_vocab: int,
        d_model: int,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.d_model: int = d_model
        self.d_vocab: int = d_vocab

    def __call__(self, x: Array) -> Array:
        # x: (batch, T, d_vocab)
        w_init: Initializer = hk.initializers.RandomNormal(
            stddev=1 / np.sqrt(self.d_vocab)
        )
        w: Array = hk.get_parameter(
            "w", shape=[self.d_vocab, self.d_model], init=w_init
        )
        # w: (d_vocab, d_model)

        # embedded_tokens: (b, T, d_model)
        return w[x, :]


def Unembed(
    d_vocab: int, with_bias: bool = False, name: Optional[str] = None
) -> hk.Module:
    return hk.Linear(
        output_size=d_vocab,
        with_bias=with_bias,
        w_init=hk.initializers.RandomNormal(stddev=1 / np.sqrt(d_vocab)),
        name=name,
    )


class LayerNorm(hk.Module):
    """
    We normalize all of the inputs for a single embedding by mean & variance
    we then scale and bias each dimension individually using learned parameters
    w_learned and b_learned
    """

    def __init__(self, epsilon: float = 1e-4, name: Optional[str] = None):
        super().__init__(name)
        self.epsilon: float = epsilon

    def __call__(self, x: Array) -> Array:
        # x: (..., d_model)
        d_model: int = x.shape[-1]

        w_learned: Array = hk.get_parameter(
            "w", shape=[d_model], init=hk.initializers.Constant(1.0)
        )
        b_learned: Array = hk.get_parameter(
            "b", shape=[d_model], init=hk.initializers.Constant(0.0)
        )

        mean: Array = x.mean(axis=-1)[..., None]
        std: Array = x.std(axis=-1)[..., None]
        # mean, std: (..., 1)

        x = (x - mean) / (std + self.epsilon)
        x = (x * w_learned) + b_learned

        return x


class FusedSelfAttention(hk.Module):
    def __init__(self, n_heads: int, name: Optional[str] = None):
        super().__init__(name)
        self.n_heads: int = n_heads

    def __call__(self, x: Array) -> Array:
        # x: (b, T, d)
        *_, seq_len, d_model = x.shape

        w_qkv: Array = hk.get_parameter(
            "w_qkv",
            shape=[d_model, 3 * d_model],
            init=hk.initializers.RandomNormal(stddev=1 / np.sqrt(d_model)),
        )
        # w_qkv: (d, 3 * d)

        qkv: Array = x @ w_qkv
        # qkv: (b, T, 3 * d)

        qkv = jnp.reshape(
            qkv, (*qkv.shape[:-1], self.n_heads, 3 * d_model // self.n_heads)
        )
        # qkv: (b, T, h, 3 * d // h)

        qkv = jnp.einsum("...Thd->...hTd", qkv)
        # qkv: (b, h, T, 3 * d // h)

        q, k, v = jnp.split(qkv, 3, axis=-1)
        # q, k, v: (b, h, T, d // h)

        # TODO: swap to k
        raw_attn: Array = q @ jnp.swapaxes(k, -2, -1)
        # raw_attn: (b, h, T, T)
        raw_attn /= np.sqrt(d_model)

        mask: Array = jnp.tri(seq_len)
        # (T, T)
        masked_attn: Array = jnp.where(mask, raw_attn, -1e30)
        # (b, h, T, T) -> broadcasting on the first two dims

        attn: Array = nn.softmax(masked_attn, axis=-1)
        # (b, h, T, T)

        attn_vals: Array = attn @ v
        # (b, h, T, d // h)

        attn_vals = jnp.swapaxes(attn_vals, -3, -2)
        # (b, T, h, d // h)
        attn_vals = jnp.reshape(attn_vals, (*attn_vals.shape[:-2], -1))
        # (b, T, d)

        w_o: Array = hk.get_parameter(
            "w_o",
            shape=[d_model, d_model],
            init=hk.initializers.RandomNormal(stddev=1 / np.sqrt(d_model)),
        )
        # (d, d)

        # (b, T, d)
        return attn_vals @ w_o


class ReLU(hk.Module):
    def __call__(self, x: Array) -> Array:
        return nn.relu(x)


class TransformerBlock(hk.Module):
    def __init__(
        self, n_heads: int, n_mlp_nodes: int, d_model: int, name: Optional[str] = None
    ):
        super().__init__(name)
        self.self_attn: hk.Module = FusedSelfAttention(n_heads, "self_attn")
        self.mlp: hk.Module = hk.Sequential(
            [hk.Linear(n_mlp_nodes), ReLU(), hk.Linear(d_model)]
        )

    def __call__(self, x: Array) -> Array:
        attn_output: Array = self.self_attn(x)
        return x + self.mlp(x + attn_output)
