#ifndef SRC_LAYER_H_
#define SRC_LAYER_H_

#include <functional>
#include <optional>
#include <utility>

#include "assert.h"
#include "activation.h"
#include "config.h"
#include "cost.h"
#include "matrix.h"

class Layer {
 public:
  explicit Layer(Config cfg, Matrix weights, Matrix biases) :
    cfg_(std::move(cfg)),
    weights_(std::move(weights)),
    biases_(std::move(biases)) {
      ASSERT(weights_.ColCount() == biases_.ColCount());
      ASSERT(biases_.RowCount() == 1);
    }

  struct LearnCache {
    Layer* layer;
    Matrix input;
    Matrix w_input;
    Matrix activated;
    std::optional<Matrix> pd_cost_weighted_input;
  };

  int32_t InputSize();
  int32_t OutputSize();

  Matrix FeedForward(Matrix input, LearnCache* cache);

  void CalcPDCostWeightedInputOutput(LearnCache* cache, Matrix expected_output);
  void CalcPDCostWeightedInputIntermed(LearnCache* cache, LearnCache* next_cache);
  void FinishBackPropagate(LearnCache* cache);

 private:
  Config cfg_;
  Matrix weights_;
  Matrix biases_;
};

#endif
