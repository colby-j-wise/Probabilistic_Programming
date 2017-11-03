import tensorflow as tf
import numpy as np
import edward as ed


def set_random_seeds(x):
    tf.set_random_seed(x)
    ed.set_seed(x)
    np.random.seed(x)
