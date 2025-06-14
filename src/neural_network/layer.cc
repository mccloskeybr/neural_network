#include "src/neural_network/layer.h"

#include <optional>

#include "absl/log/check.h"
#include "src/neural_network/activation.h"
#include "src/neural_network/cost.h"
#include "src/neural_network/params.h"

int32_t Layer::InputSize() const { return weights_.RowCount(); }

int32_t Layer::OutputSize() const { return weights_.ColCount(); }

const Matrix& Layer::Weights() const { return weights_; }

const Matrix& Layer::Biases() const { return biases_; }

Matrix Layer::Infer(const Matrix& input) const {
  return GetActivation(activation_)(input * weights_);
}

Matrix Layer::FeedForward(const Matrix& input, LayerLearnCache* cache) const {
  const Matrix w_input = (input * weights_);
  const Matrix activated = GetActivation(activation_)(w_input);
  *cache = LayerLearnCache {
    .layer = this,
    .input = input,
    .w_input = w_input,
    .activated = activated,
    .pd_cost_weighted_input = std::nullopt,
  };
  return activated;
}

void Layer::CalcPDCostWeightedInputOutput(
    const TrainParameters& train_params, LayerLearnCache* cache, const Matrix& expected_output) const {
  const Matrix pd_cost_activation = GetCostDeriv(train_params.cost)(cache->activated, expected_output);
  Matrix pd_activation_weighted_input = GetActivationDeriv(activation_)(cache->w_input);
  pd_activation_weighted_input.HadamardMultInPlace(pd_cost_activation);
  cache->pd_cost_weighted_input = std::move(pd_activation_weighted_input);
}

void Layer::CalcPDCostWeightedInputIntermed(LayerLearnCache* cache, LayerLearnCache* next_cache) const {
  DCHECK(next_cache->pd_cost_weighted_input.has_value());
  // SPEEDUP: Matrix.MultTranspose
  cache->pd_cost_weighted_input =
    *next_cache->pd_cost_weighted_input * next_cache->layer->weights_.Transpose();
  cache->pd_cost_weighted_input->HadamardMultInPlace(GetActivationDeriv(activation_)(cache->w_input));
}

std::pair<Matrix, Matrix> Layer::FinishBackPropagate(LayerLearnCache* cache) const {
  DCHECK(cache->pd_cost_weighted_input.has_value());
  // SPEEDUP: Matrix.TransposeMult
  Matrix cost_gradient_weights =
    cache->input.Transpose() * *cache->pd_cost_weighted_input;
  Matrix& cost_gradient_biases =
    *cache->pd_cost_weighted_input /* * 1.0 */;
  return std::make_pair(cost_gradient_weights, cost_gradient_biases);
}

void Layer::ApplyGradients(const TrainParameters& train_params, std::pair<Matrix, Matrix> gradients) {
  Matrix cost_gradient_weights = std::move(gradients.first);
  double weight_decay = (1.0 - train_params.regularization * train_params.learn_rate);
  cost_gradient_weights *= train_params.learn_rate;
  weight_velocities_ *= train_params.momentum;
  weight_velocities_ -= cost_gradient_weights;
  weights_ *= weight_decay;
  weights_ += weight_velocities_;

  Matrix cost_gradient_biases = std::move(gradients.second);
  cost_gradient_biases *= train_params.learn_rate;
  bias_velocities_ *= train_params.momentum;
  bias_velocities_ -= cost_gradient_biases;
  biases_ += bias_velocities_;
}
