from typing import Any, NamedTuple, Optional, Tuple

import haiku as hk
import jax
import jax.nn as nn
import jax.numpy as jnp
import numpy as np
import optax
from chex import ArrayNumpy
from haiku.initializers import Initializer
from jax import Array
from jax.random import KeyArray, PRNGKey
from optax import GradientTransformation, OptState, Params

FrozenValues = dict[str, Any]


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

    def _qkv(self, x: Array, d_model: int, frozen: FrozenValues) -> dict[str, Array]:
        """
        Parameters
        ----------
        x : (b, T, d)
        cache: {
            "q"?: (b, h, T, d // h)
            "k"?: (b, h, T, d // h)
            "v"?: (b, h, T, d // h)
        }
        Returns
        -------
        {
            "q": (b, h, T, d // h)
            "k": (b, h, T, d // h)
            "v": (b, h, T, d // h)
        }

        """

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

        return {
            "q": frozen.get("q", q),
            "k": frozen.get("k", k),
            "v": frozen.get("v", v),
        }

    def _attn(
        self, q: Array, k: Array, d_model: int, frozen: FrozenValues
    ) -> dict[str, Array]:
        """
        Parameters
        ----------
        q: (b, h, T, d // h)
        k: (b, h, T, d // h)
        cache: {
            "attn_logits"?: (b, h, T, T),
            "mask"?: (T, T),
            "masked_attn"?: (b, h, T, T)
        }
        Returns
        -------
        {
            "attn_logits": (b, h, T, T),
            "mask": (T, T),
            "masked_attn": (b, h, T, T)
        }

        """
        attn_logits: Array = frozen.get(
            "attn_logits", q @ jnp.swapaxes(k, -2, -1) / np.sqrt(d_model)
        )
        # raw_attn: (b, h, T, T)

        *_, seq_len = attn_logits.shape
        mask: Array = frozen.get("mask", jnp.tri(seq_len))
        # (T, T)

        masked_attn: Array = jnp.where(mask, attn_logits, -1e30)
        # (b, h, T, T) -> broadcasting on the first two dims

        attn: Array = nn.softmax(masked_attn, axis=-1)
        # (b, h, T, T)

        return {
            "attn_logits": attn_logits,
            "mask": mask,
            "attn": frozen.get("attn", attn),
        }

    def __call__(self, x: Array, frozen: FrozenValues) -> FrozenValues:
        """
        Parameters
        ----------
        x : (b, T, d)
        Returns
        -------
        """
        *_, d_model = x.shape

        qkv: dict[str, Array] = self._qkv(x, d_model, frozen)
        attn: dict[str, Array] = self._attn(qkv["q"], qkv["k"], d_model, frozen)

        attn_vals: Array = attn["attn"] @ qkv["v"]
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
        return {"out": frozen.get("out", attn_vals @ w_o), **qkv, **attn}


class ReLU(hk.Module):
    def __call__(self, x: Array) -> Array:
        return nn.relu(x)


class MLP(hk.Module):
    def __init__(self, d_mlp_nodes: int, d_out: int, name: Optional[str] = None):
        super().__init__(name)
        self.linear1 = hk.Linear(d_mlp_nodes)
        self.linear2 = hk.Linear(d_out)

    def __call__(self, x: Array, frozen: FrozenValues) -> FrozenValues:
        result: FrozenValues = {}

        result["neurons"] = frozen.get("neurons", self.linear1(x))
        result["relu_neurons"] = frozen.get("relu_neurons", nn.relu(result["neurons"]))
        result["out"] = frozen.get("out", self.linear2(result["relu_neurons"]))

        return result


class TransformerBlock(hk.Module):
    def __init__(
        self, n_heads: int, d_mlp_nodes: int, d_model: int, name: Optional[str] = None
    ):
        super().__init__(name)
        self.self_attn = FusedSelfAttention(n_heads, "self_attn")
        self.mlp = MLP(d_mlp_nodes, d_model, "mlp")

    def __call__(self, x: Array, frozen: FrozenValues) -> FrozenValues:
        result: FrozenValues = {}

        result["self_attn"] = self.self_attn(x, frozen.get("self_attn", {}))
        result["pre_mlp_residual"] = frozen.get(
            "pre_mlp_residual", x + result["self_attn"]["out"]
        )
        result["mlp"] = self.mlp(
            result["pre_mlp_residual"], frozen=frozen.get("mlp", {})
        )
        result["out"] = frozen.get("out", x + result["mlp"]["out"])

        return result
