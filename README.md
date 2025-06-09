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
* Feature: revisit neural net calculus for potential use of proper linear algebra
* Speedup: matrix operation performance boosting
* Speedup: could multithread back prop for weights vs. biases instead of both
* Speedup: centralized thread pool instead of spawning threads all the time
