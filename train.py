from time import time
from transformer import Decoder
from data import Dataloader
import jax
from jax import random
import optax
from flax import serialization


N_TRAIN = 500000
SAVE_EPOCH = 50000
LEARNING_RATE = 6e-4
BLOCK_SIZE = 32
BATCH_SIZE = 64
N_LAYERS = 3
N_EMBD = 256
HEADS = 8
N_INNER = 512
MODELS_PATH = "models"

if __name__ == "__main__":

    data = Dataloader(batch_size=BATCH_SIZE, block_size=BLOCK_SIZE)
    decoder = Decoder(
        n_layers=N_LAYERS,
        n_vocab=data.n_vocab,
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        heads=HEADS,
        n_inner=N_INNER,
    )

    key1, key2, dropout_key = random.split(random.key(0), 3)
    x = random.randint(key1, (BATCH_SIZE, BLOCK_SIZE), minval=0, maxval=data.n_vocab)
    params = decoder.init(key2, x, training=False)

    # training loss fn
    @jax.jit
    def loss_fn(params, x, y):
        logits = decoder.apply(params, x, training=True, rngs={"dropout": dropout_key})
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    

    tx = optax.adam(learning_rate=LEARNING_RATE)
    opt_state = tx.init(params)
    loss_grad_fn = jax.value_and_grad(loss_fn)

    start_time = time()
    for i in range(N_TRAIN):
        x, y = data.get_batch()
        loss, grads = loss_grad_fn(params, x, y)
        updates, opt_state = tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        if i % 100 == 0:
            time_elapsed = time() - start_time
            print(f"batch {i}, used time: {time_elapsed:.2f}, loss: {loss}")
        if i % SAVE_EPOCH == 0:
            param_bytes = serialization.to_bytes(params)
            save_path = f"{MODELS_PATH}/model_{i}.bin"
            with open(save_path, "wb") as f:
                f.write(param_bytes)
            print(f"model saved at epoch {i}")