#ifndef SRC_NEURAL_NETWORK_PARAMS_H_
#define SRC_NEURAL_NETWORK_PARAMS_H_

#include "absl/strings/str_cat.h"
#include "src/neural_network/cost.h"

struct TrainParameters {
  std::string ToString() const {
    return absl::StrCat(
        "{ cost: ", CostToString(cost),
        ", learn_rate: ", learn_rate,
        ", momentum: ", momentum,
        ", regularization: ", regularization,
        ", num_threads: ", num_threads,
        ", num_epochs: ", num_epochs,
        ", train_batch_size: ", train_batch_size,
        ", test_batch_size: ", test_batch_size,
        " }");
  }

  Cost cost;
  double learn_rate;
  double momentum;
  double regularization;
  uint32_t num_threads;
  uint32_t num_epochs;
  uint32_t train_batch_size;
  uint32_t test_batch_size;
};

#endif
