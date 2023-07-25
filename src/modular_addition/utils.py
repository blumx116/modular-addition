import os
import pickle
from typing import NamedTuple

import haiku as hk
import jax
import numpy as np
from jax import Array
from jax.random import KeyArray
from optax import OptState


class StepReturn(NamedTuple):
    loss: Array  # item
    params: hk.MutableParams
    opt_state: OptState


class BatchReturn(NamedTuple):
    x: Array
    y: Array
    rng: KeyArray


def save(checkpoint_dir: str, state) -> None:
    # stolen from https://github.com/deepmind/dm-haiku/issues/18#issuecomment-981814403
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    with open(os.path.join(checkpoint_dir, "arrays.npy"), "wb") as f:
        for x in jax.tree_util.tree_leaves(state):
            np.save(f, x, allow_pickle=False)

    tree_struct = jax.tree_map(lambda t: 0, state)
    with open(os.path.join(checkpoint_dir, "tree.pkl"), "wb") as f:
        pickle.dump(tree_struct, f)


def restore(ckpt_dir):
    # stolen from https://github.com/deepmind/dm-haiku/issues/18#issuecomment-981814403
    with open(os.path.join(ckpt_dir, "tree.pkl"), "rb") as f:
        tree_struct = pickle.load(f)

    leaves, treedef = jax.tree_util.tree_flatten(tree_struct)
    with open(os.path.join(ckpt_dir, "arrays.npy"), "rb") as f:
        flat_state = [np.load(f) for _ in leaves]

    return jax.tree_util.tree_unflatten(treedef, flat_state)
