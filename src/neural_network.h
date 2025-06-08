#ifndef SRC_NEURAL_NETWORK_H_
#define SRC_NEURAL_NETWORK_H_

#include <vector>

#include "config.h"
#include "layer.h"

class NeuralNetwork {
 public:
  explicit NeuralNetwork(
      Config cfg, std::vector<Matrix> weights, std::vector<Matrix> biases);
  static NeuralNetwork Random(
      Config cfg, std::vector<int32_t> layer_sizes);

  float Learn(Matrix input, int32_t expected_class);

 private:
  Config cfg_;
  std::vector<Layer> layers_;
};

#endif
