#ifndef SRC_PARAMS_H_
#define SRC_PARAMS_H_

#include "src/neural_network/activation.h"
#include "src/neural_network/cost.h"

struct Parameters {
  std::vector<uint32_t> layer_sizes;
  Activation intermed_activation;
  Activation output_activation;
  Cost cost;
  double learn_rate;
  double momentum;
  double regularization;
  uint32_t num_threads;
  uint32_t batch_size;
  uint32_t num_epochs;
};

#endif
