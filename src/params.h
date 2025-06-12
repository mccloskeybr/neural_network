#ifndef SRC_PARAMS_H_
#define SRC_PARAMS_H_

#include "src/neural_network/activation.h"
#include "src/neural_network/cost.h"

struct Parameters {
  std::vector<uint32_t> layer_sizes;
  Activation activation;
  Cost cost;
  double learn_rate;
  uint32_t num_threads;
  uint32_t batch_size;
  uint32_t num_epochs;
};

#endif
