#ifndef SRC_LAYER_H_
#define SRC_LAYER_H_

#include <functional>
#include <optional>
#include <utility>

#include "src/params.h"
#include "src/common/assert.h"
#include "src/common/matrix.h"
#include "src/neural_network/activation.h"
#include "src/neural_network/cost.h"

class Layer {
 public:
  explicit Layer(Parameters* params, Matrix weights, Matrix biases) :
    params_(params),
    weights_(std::move(weights)),
    biases_(std::move(biases)) {
      ASSERT(weights_.ColCount() == biases_.ColCount());
      ASSERT(biases_.RowCount() == 1);
    }

  struct LayerLearnCache {
    const Layer* layer;
    Matrix input;
    Matrix w_input;
    Matrix activated;
    std::optional<Matrix> pd_cost_weighted_input;
  };

  int32_t InputSize() const;
  int32_t OutputSize() const;

  Matrix FeedForward(Matrix input, LayerLearnCache* cache) const;

  void CalcPDCostWeightedInputOutput(LayerLearnCache* cache, Matrix expected_output) const;
  void CalcPDCostWeightedInputIntermed(LayerLearnCache* cache, LayerLearnCache* next_cache) const;
  std::pair<Matrix, Matrix> FinishBackPropagate(LayerLearnCache* cache) const;
  void ApplyGradients(std::pair<Matrix, Matrix> gradients);

 private:
  Parameters* params_;
  Matrix weights_;
  Matrix biases_;
};

#endif
