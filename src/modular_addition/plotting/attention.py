from typing import Protocol, Tuple

import haiku as hk
import jax.numpy as jnp
import plotly.express as px
import plotly.graph_objects as go
from jax import Array
from jax.random import KeyArray, PRNGKey
from plotly.offline.offline import plot_mpl

from modular_addition.model import FrozenValues
from modular_addition.train import P, def_generate_batch, model
from modular_addition.utils import restore

params = restore("checkpoints/modular-addition/wd-0.1-tf-0.7/checkpoint-40000/")


rng: KeyArray = PRNGKey(777)

gen_train_batch, gen_test_bach = def_generate_batch(
    rng, train_frac=0.9, no_symmetry_leak=True
)


class ForwardFunction(Protocol):
    def __call__(
        self, rng: KeyArray, params: hk.MutableParams, x: Array, frozen: FrozenValues
    ) -> Tuple[Array, FrozenValues]:
        ...


def run_on_all_pairs(params: hk.MutableParams, fw: ForwardFunction) -> FrozenValues:
    inputs: Array = jnp.asarray([(i, j, P) for i in range(P) for j in range(P)])
    _, frozen = fw(rng=PRNGKey(0), params=params, x=inputs, frozen={})
    # every matrix should have b=P**2
    return frozen


def plot_attn_pct_at_0(frozen: FrozenValues) -> go.Figure:
    attn: Array = frozen["representation"]["self_attn"]["attn"]
    attn_pct_at_0: Array = attn[:, :, 2, 0]
    # (P ** 2, h)
    # attn paid from '2' (the '=' sign) to first number

    attn_pct_at_0: Array = attn_pct_at_0.reshape(P, P, -1)

    return px.imshow(attn_pct_at_0, animation_frame=2)


def plot_mean_attn_at_0(frozen: FrozenValues) -> go.Figure:
    attn: Array = frozen["representation"]["self_attn"]["attn"]
    return px.imshow(attn[:, :, 2, :].mean(0))


def plot_mlp_outputs(frozen: FrozenValues) -> go.Figure:
    mlp_outputs: Array = frozen["representation"]["out"][:, -1, :]
    # (P **2, d)
    return px.imshow(mlp_outputs.reshape(P, P, -1), animation_frame=2)


def fourier_basis(p: int) -> Array:
    basis: list[Array] = []

    for i in range(1, (p // 2) + 1):
        for fn in [jnp.cos, jnp.sin]:
            unnormalized_basis: Array = fn(2 * jnp.pi * i / p * jnp.arange(p))
            # (p, )
            basis.append(unnormalized_basis / jnp.linalg.norm(unnormalized_basis))

    return jnp.stack(basis, axis=0)


def fourier_basis_names(p: int) -> list[str]:
    basis_names: list[str] = []

    for i in range(1, (p // 2) + 1):
        for fn_name in ["cos", "sine"]:
            basis_names.append(f"{fn_name} {i}")

    return basis_names


def fourier_basis_term_2d(basis: Array, i: int, j: int) -> Array:
    v_i: Array = basis[i, :][:, None]
    v_j: Array = basis[j, :][None, :]

    return v_i @ v_j


train_batch: Array = gen_train_batch(rng, batch_size=32)

_, fw = hk.transform(model)
frozen: FrozenValues = run_on_all_pairs(params, fw)

plot_mlp_outputs(frozen).show()

basis: Array = fourier_basis(69)
