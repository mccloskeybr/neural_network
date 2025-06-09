#ifndef SRC_PARAMS_H_
#define SRC_PARAMS_H_

#include "src/neural_network/activation.h"
#include "src/neural_network/cost.h"

struct Parameters {
  std::vector<uint32_t> layer_sizes;
  Activation activation = Activation::SIGMOID;
  Cost cost = Cost::MEAN_SQUARED;
  float learn_rate = 0.05;
  int32_t num_threads = 32;
  int32_t batch_size = 30;
  int32_t num_epochs = 1;
};

#endif
