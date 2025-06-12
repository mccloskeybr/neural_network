# Neural Network

Playing around with custom neural network architecture, for fun.

Feature set:
* Can define arbitrary network shapes.
* Can define arbitrary activation / cost functions.
* Support for epoch / batch based training.
* Batches are multithreaded.
* Easy to play around with different hyper parameters.

TODO:
* Feature: non-static learning rate, momentum
* Speedup: matrix operation performance boosting
* Speedup: could multithread back prop for weights vs. biases instead of both
