# Neural Network

Playing around with neural network architecture, for fun. Tested on the MNIST and MNIST-fashion datasets.

See `run.bat` for an example of how to run the program.

Feature set:
* Implementation support for backpropogation / gradient descent based learning.
* Support for reading / writing model checkpoints.
* Can define arbitrary network shapes.
* Can define arbitrary activation / cost functions.
* Support for epoch / batch based training.
* Training and inference batches are multithreaded to maximize system resources.
* Easy to play around with different hyper parameters.

Dependencies:
* Built using bazel.
* Model checkpoints are stored in protobuf format.
* Uses the absl and googletest libraries.
