#ifndef SRC_CONFIG_H_
#define SRC_CONFIG_H_

#include "src/neural_network/activation.h"
#include "src/neural_network/cost.h"

struct Config {
  Activation activation;
  Cost cost;
  float learn_rate;
};

#endif
