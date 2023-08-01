import haiku as hk
import plotly.express as px
from jax import Array

from modular_addition.train import model
from modular_addition.utils import restore

weights: hk.MutableParams = restore(
    "checkpoints/modular-addition/wd-0.01-tf-0.7/checkpoint-5000/"
)

embedding_matrix: Array = weights["embeddings"]["w"][:, 17:18]

px.line(embedding_matrix).show()
