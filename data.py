from typing import Dict, Any, Tuple
import os
import numpy as np
import jax.numpy as jnp


class Dataloader():
    """
    Simple dataloader that loads the tiny shakespeare dataset
    """

    def __init__(self, batch_size: int, block_size: int) -> None:
        if not os.path.exists('input.txt'):
            os.system('wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')
        
        text = open('input.txt', 'r').read()
        self.vocab = sorted(list(set(text)))
        self.n_vocab = len(self.vocab)
        
        self.stoi = {c: i for i, c in enumerate(self.vocab)}
        self.itos = {i: c for c, i in self.stoi.items()}
        
        self.tokens = [self.stoi[c] for c in text]

        self.batch_size = batch_size
        self.block_size = block_size
        
    
    def get_batch(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        start_inds = np.random.randint(0, len(self.tokens) - self.block_size, self.batch_size)
        x = np.stack([self.tokens[start:start + self.block_size] for start in start_inds])
        y = np.stack([self.tokens[start + 1:start + self.block_size + 1] for start in start_inds])
        return jnp.asarray(x), jnp.asarray(y)