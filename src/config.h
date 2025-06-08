#ifndef SRC_CONFIG_H_
#define SRC_CONFIG_H_

#include "activation.h"
#include "cost.h"

struct Config {
  Activation activation;
  Cost cost;
  float learn_rate;
};

#endif
