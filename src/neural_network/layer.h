#ifndef SRC_LAYER_H_
#define SRC_LAYER_H_

#include <functional>
#include <optional>
#include <utility>

#include "src/common/assert.h"
#include "src/common/matrix.h"
#include "src/neural_network/activation.h"
#include "src/neural_network/cost.h"

class Layer {
 public:
  explicit Layer(
      Matrix weights, Matrix biases,
      Activation activation, Cost cost,
      double learn_rate, double momentum,
      double regularization) :
    weights_(std::move(weights)),
    biases_(std::move(biases)),
    weight_velocities_(Matrix(weights_.RowCount(), weights_.ColCount())),
    bias_velocities_(Matrix(biases_.RowCount(), biases_.ColCount())),
    activation_(activation),
    cost_(cost),
    learn_rate_(learn_rate),
    momentum_(momentum),
    regularization_(regularization) {
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
  const Matrix& Weights() const;
  const Matrix& Biases() const;

  Matrix FeedForward(const Matrix& input, LayerLearnCache* cache) const;

  void CalcPDCostWeightedInputOutput(LayerLearnCache* cache, const Matrix& expected_output) const;
  void CalcPDCostWeightedInputIntermed(LayerLearnCache* cache, LayerLearnCache* next_cache) const;
  std::pair<Matrix, Matrix> FinishBackPropagate(LayerLearnCache* cache) const;
  void ApplyGradients(const std::pair<Matrix, Matrix>& gradients);

 private:
  Matrix weights_;
  Matrix biases_;
  Matrix weight_velocities_;
  Matrix bias_velocities_;
  Activation activation_;
  Cost cost_;
  double learn_rate_;
  double momentum_;
  double regularization_;
};

#endif
