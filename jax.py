# Neural netwrok in Jax, ripped from: https://docs.jax.dev/en/latest/notebooks/Neural_Network_and_Data_Loading.html

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.scipy.special import logsumexp
import numpy as np
from jax.tree_util import tree_map
from torch.utils.data import DataLoader, default_collate
from torchvision.datasets import MNIST
import time

def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

def relu(x):
  return jnp.maximum(0, x)

def predict(network, image):
    activations = image
    for w, b in network[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)

    final_w, final_b = network[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return logits - logsumexp(logits)

batched_predict = vmap(predict, in_axes=(None, 0))

def one_hot(x, k, dtype=jnp.float32):
    return jnp.array(x[:, None] == jnp.arange(k), dtype)

def accuracy(network, images, targets):
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(batched_predict(network, images), axis=1)
    return jnp.mean(predicted_class == target_class)

def loss(network, images, targets):
    preds = batched_predict(network, images)
    return -jnp.mean(preds * targets)

@jit
def update(network, x, y):
    grads = grad(loss)(network, x, y)
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(network, grads)]

def numpy_collate(batch):
    return tree_map(np.asarray, default_collate(batch))

def flatten_and_cast(pic):
    """Convert PIL image to flat (1-dimensional) numpy array."""
    return np.ravel(np.array(pic, dtype=jnp.float32))

layer_sizes = [784, 512, 512, 10]
step_size = 0.01
num_epochs = 10
batch_size = 128
n_targets = 10
network = init_network_params(layer_sizes, random.key(0))

mnist_dataset = MNIST('/tmp/mnist/', download=True, transform=flatten_and_cast)
training_generator = DataLoader(mnist_dataset, batch_size=batch_size, collate_fn=numpy_collate)

train_images = np.array(mnist_dataset.train_data).reshape(len(mnist_dataset.train_data), -1)
train_labels = one_hot(np.array(mnist_dataset.train_labels), n_targets)

mnist_dataset_test = MNIST('/tmp/mnist/', download=True, train=False)
test_images = jnp.array(mnist_dataset_test.test_data.numpy().reshape(len(mnist_dataset_test.test_data), -1), dtype=jnp.float32)
test_labels = one_hot(np.array(mnist_dataset_test.test_labels), n_targets)

for epoch in range(num_epochs):
    start_time = time.time()
    for input, expected_class in training_generator:
        expected_output = one_hot(expected_class, n_targets)
        network = update(network, input, expected_output)
    epoch_time = time.time() - start_time

    train_acc = accuracy(network, train_images, train_labels)
    test_acc = accuracy(network, test_images, test_labels)

    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    print("Training set accuracy {}".format(train_acc))
    print("Test set accuracy {}".format(test_acc))
