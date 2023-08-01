import haiku as hk
import jax.numpy as jnp
from jax import Array

from modular_addition.utils import restore

params: hk.MutableParams = restore(
    "checkpoints/modular-addition/wd-0.1-tf-0.7/checkpoint-40000/"
)
import pdb

pdb.set_trace()
print()

W_e: Array = params["embeddings"]["w"]
W_o: Array = params["transformer/~/self_attn"]["w_o"]
_, _, W_v = jnp.split(params["transformer/~/self_attn"]["w_qkv"], 3, axis=-1)

import pdb

pdb.set_trace()
print()
