import pickle
from functools import partial
from itertools import product
from typing import Callable, NamedTuple, Optional, Tuple

import haiku as hk
import jax
import jax.nn as nn
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import plotly.express as px
from haiku.initializers import Initializer
from jax import Array
from jax.random import KeyArray, PRNGKey
from optax import GradientTransformation, OptState, Params
from tqdm import tqdm

from modular_addition.model import Embeddings, FrozenValues, TransformerBlock, Unembed
from modular_addition.utils import BatchReturn, StepReturn, save

N_EPOCHS: int = 50000
P: int = 69
BATCH_SIZE: int = 512
D_VOCAB: int = P + 1
N_HEADS: int = 4
N_MLP_NODES: int = 512
D_MODEL: int = 128
ETA: float = 3e-4


def model(x: Array, frozen: FrozenValues) -> Tuple[Array, FrozenValues]:
    result: FrozenValues = {}
    # x: (b, T) # int IDs

    result["embedding"] = frozen.get(
        "embedding", Embeddings(D_VOCAB, D_MODEL, name="embeddings")(x)
    )
    # embedding: (b, T, d)

    result["representation"] = TransformerBlock(
        N_HEADS, N_MLP_NODES, D_MODEL, name="transformer"
    )(result["embedding"], frozen.get("representation", {}))
    # representation: (b, T, d)

    result["out"] = frozen.get(
        "unembeddings",
        Unembed(D_VOCAB, name="unembeddings")(result["representation"]["out"]),
    )
    # result: (b, T, vocab)
    return result["out"], result


def loss(logits: Array, targets: Array) -> Array:
    # targets: (b, T, vocab)
    # logits: (b, T, vocab)

    targets: Array = nn.one_hot(targets, D_VOCAB)
    # (b, vocab)
    # TODO: I think that we don't need to explicitly handle masking
    # because targets outside of the range of [0, D_VOCAB] (i.e. -1)
    # are encoded as all 0s, which means that they won't contribute to the sum
    # below

    loss: Array = -jnp.sum(targets * nn.log_softmax(logits, axis=-1)) / targets.sum()

    return loss


def forward_with_loss(x: Array, y: Array) -> Array:
    # x: (b, T) # int ids
    # y: (b, T) # int ids
    #   NOTE: -1 is used as convention for masked loss
    logits, _ = model(x, frozen={})
    # preds: (b, T, vocab)
    # ignore all but last timestep prediction
    return loss(logits, y)


def _generate_batch(
    rng: KeyArray, options: list[tuple[int, int]], batch_size: Optional[int] = None
) -> BatchReturn:
    batch_size = batch_size or len(options)
    assert batch_size <= len(options)
    rng, tmp_rng = jax.random.split(rng)
    x: Array = jax.random.permutation(tmp_rng, options)[:batch_size]
    # x: (b, 2)
    y: Array = jnp.mod(x.sum(axis=-1), P)
    # y: (b, )
    y = y[:, None]
    # y: (b, 1)

    y = jnp.concatenate([jnp.full_like(x, -1), y], axis=1)
    x = jnp.concatenate([x, jnp.full((batch_size, 1), fill_value=P)], axis=1)
    # x, y: (b, 3)

    return BatchReturn(x, y, rng)


def _clean_symmetry_leak(data: list[tuple[int, int]]) -> list[tuple[int, int]]:
    # the transformer will be inherently permutation invariant without positional embeddings
    # this is because the only output that we actually pay attention to is the last token
    # and it pays attention to both of the first two tokens in the same way
    # neither of the first two tokens are affected by each other at that point, because this is a
    # one-layer network

    # as a result, this function changes the data so that both (i, j) and (j, i) are guaranteed
    # to be in the same split of the dataset
    # we do this by first removing all instances where j>i, and then re-adding them to the dataset
    # where i>j is found
    data = [(i, j) for (i, j) in data if j <= i]
    reflected_data: list[tuple[int, int]] = [(j, i) for (i, j) in data]
    data = data + reflected_data
    return list(set(reflected_data))


def def_generate_batch(
    rng: KeyArray, train_frac: float, no_symmetry_leak: bool = False
) -> Tuple[Callable[[KeyArray], BatchReturn], ...]:
    train_data: list[tuple[int, int]] = []
    test_data: list[tuple[int, int]] = []

    for i in range(P):
        for j in range(P):
            rng, tmp_rng = jax.random.split(rng)
            if jax.random.uniform(tmp_rng) < train_frac:
                train_data.append((i, j))
            else:
                test_data.append((i, j))

    if no_symmetry_leak:
        _clean_symmetry_leak(train_data)
        _clean_symmetry_leak(test_data)

    train_data: Array = jnp.asarray(train_data)
    test_data: Array = jnp.asarray(test_data)
    # (n_data_points, 2)

    return partial(_generate_batch, options=train_data), partial(
        _generate_batch, options=test_data
    )


def should_save(epoch: int) -> bool:
    # save whenever all of the digits except the leading digit are 0
    # this results in saving exponentially less and less often
    digits: str = str(epoch)
    trailing_digits: str = digits[1:]
    return trailing_digits == ("0" * len(trailing_digits))


def fit(weight_decay: float, train_frac: float) -> tuple[list[float], list[float]]:
    @jax.jit
    def step(
        params: hk.MutableParams, opt_state: OptState, x: Array, y: Array
    ) -> StepReturn:
        loss, grads = jax.value_and_grad(apply)(params, x=x, y=y, rng=rng)
        # loss : value
        # grads : hk.MutableParams
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)  # type: ignore

        return StepReturn(loss, params, opt_state)

    rng: KeyArray = PRNGKey(777)
    rng, rng_tmp = jax.random.split(rng)

    generate_train_batch, generate_test_batch = def_generate_batch(
        rng_tmp, train_frac=train_frac, no_symmetry_leak=True
    )

    xdummy, ydummy, rng = generate_train_batch(rng)
    init, apply = hk.transform(forward_with_loss)
    params: hk.MutableParams = init(rng=rng, x=xdummy, y=ydummy)

    optimizer: GradientTransformation = optax.chain(
        optax.add_decayed_weights(weight_decay * ETA), optax.adam(learning_rate=ETA)
    )
    opt_state = optimizer.init(params)

    train_losses: list[float] = []
    test_losses: list[float] = []

    for i in range(N_EPOCHS):
        try:
            if should_save(i):
                save(
                    f"checkpoints/modular-addition/wd-{weight_decay}-tf-{train_frac}/checkpoint-{i}",
                    params,
                )
            x, y, rng = generate_train_batch(rng)
            loss_value, params, opt_state = step(params, opt_state, x, y)

            xtest, ytest, rng = generate_test_batch(rng)
            losstest = apply(params=params, x=xtest, y=ytest, rng=rng)
            test_losses.append(float(losstest))
            train_losses.append(float(loss_value))
        except KeyboardInterrupt:
            df: pd.DataFrame = pd.DataFrame(
                {"train": train_losses, "test": test_losses}
            )
            fig = px.line(df, title=f"{weight_decay=}, {train_frac=}")
            fig.show()
            import pdb

            pdb.set_trace()
            print()
    return train_losses, test_losses


if __name__ == "__main__":
    all_train_losses: dict[tuple[float, float], list[float]] = {}
    all_test_losses: dict[tuple[float, float], list[float]] = {}

    for weight_decay, train_frac in tqdm(
        list(product([0, 0.01, 0.05, 0.1, 0.2], np.arange(0.95, 0, -0.05)))
    ):
        train_frac = round(train_frac, 2)  # solve stupid floating point errors
        train_losses, test_losses = fit(weight_decay, train_frac)
        all_train_losses[(weight_decay, train_frac)] = train_losses
        all_test_losses[(weight_decay, train_frac)] = test_losses

    with open("all_train_losses.pkl", "wb") as f:
        pickle.dump(all_train_losses, f)

    with open("all_test_losses.pkl", "wb") as f:
        pickle.dump(all_test_losses, f)

    import pdb

    pdb.set_trace()
    print()
