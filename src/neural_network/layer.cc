#include "src/neural_network/layer.h"

#include <optional>

#include "src/common/assert.h"
#include "src/neural_network/activation.h"
#include "src/neural_network/cost.h"

int32_t Layer::InputSize() const { return weights_.RowCount(); }

int32_t Layer::OutputSize() const { return weights_.ColCount(); }

Matrix Layer::FeedForward(const Matrix& input, LayerLearnCache* cache) const {
  const Matrix w_input = (input * weights_);
  const Matrix activated = GetActivation(activation_)(w_input);
  if (cache != nullptr) {
    *cache = LayerLearnCache {
      .layer = this,
      .input = input,
      .w_input = w_input,
      .activated = activated,
      .pd_cost_weighted_input = std::nullopt,
    };
  }
  return activated;
}

void Layer::CalcPDCostWeightedInputOutput(LayerLearnCache* cache, const Matrix& expected_output) const {
  const Matrix pd_cost_activation = GetCostDeriv(cost_)(cache->activated, expected_output);
  Matrix pd_activation_weighted_input = GetActivationDeriv(activation_)(cache->w_input);
  pd_activation_weighted_input.HadamardMultInPlace(pd_cost_activation);
  cache->pd_cost_weighted_input = std::move(pd_activation_weighted_input);
}

void Layer::CalcPDCostWeightedInputIntermed(LayerLearnCache* cache, LayerLearnCache* next_cache) const {
  ASSERT(next_cache->pd_cost_weighted_input.has_value());
  // SPEEDUP: Matrix.MultTranspose
  cache->pd_cost_weighted_input =
    *next_cache->pd_cost_weighted_input * next_cache->layer->weights_.Transpose();
  cache->pd_cost_weighted_input->HadamardMultInPlace(GetActivationDeriv(activation_)(cache->w_input));
}

std::pair<Matrix, Matrix> Layer::FinishBackPropagate(LayerLearnCache* cache) const {
  ASSERT(cache->pd_cost_weighted_input.has_value());
  // SPEEDUP: Matrix.TransposeMult
  Matrix cost_gradient_weights =
    cache->input.Transpose() * *cache->pd_cost_weighted_input;
  Matrix& cost_gradient_biases =
    *cache->pd_cost_weighted_input /* * 1.0 */;
  return std::make_pair(cost_gradient_weights, cost_gradient_biases);
}

void Layer::ApplyGradients(const std::pair<Matrix, Matrix>& gradients) {
  const Matrix& cost_gradient_weights = gradients.first;
  double weight_decay = (1.0 - regularization_ * learn_rate_);
  weight_velocities_ =
    (weight_velocities_ * momentum_) -
    (cost_gradient_weights * learn_rate_);
  weights_ = weights_ * weight_decay + weight_velocities_;

  const Matrix& cost_gradient_biases = gradients.second;
  bias_velocities_ =
    (bias_velocities_ * momentum_) -
    (cost_gradient_biases * learn_rate_);
  biases_ = biases_ + bias_velocities_;
}
