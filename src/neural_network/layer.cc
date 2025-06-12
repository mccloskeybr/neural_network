#include "src/neural_network/layer.h"

#include <optional>

#include <iostream>
#include <ostream>

#include "src/common/assert.h"
#include "src/neural_network/activation.h"
#include "src/neural_network/cost.h"

int32_t Layer::InputSize() const { return weights_.RowCount(); }

int32_t Layer::OutputSize() const { return weights_.ColCount(); }

Matrix Layer::FeedForward(const Matrix input, LayerLearnCache* cache) const {
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

void Layer::CalcPDCostWeightedInputOutput(LayerLearnCache* cache, const Matrix expected_output) const {
  const Matrix pd_cost_activation = cache->activated.Merge(expected_output, GetCostDeriv(cost_));
  const Matrix pd_activation_weighted_input = GetActivationDeriv(activation_)(cache->w_input);
  cache->pd_cost_weighted_input = pd_cost_activation.HadamardMult(pd_activation_weighted_input);
}

void Layer::CalcPDCostWeightedInputIntermed(LayerLearnCache* cache, LayerLearnCache* next_cache) const {
  ASSERT(next_cache->pd_cost_weighted_input.has_value());
  cache->pd_cost_weighted_input = Matrix(1, next_cache->layer->weights_.RowCount());
  for (int32_t i = 0; i < next_cache->layer->weights_.RowCount(); i++) {
    double pd_cost_weighted_input = 0.0;
    for (int32_t j = 0; j < next_cache->layer->weights_.ColCount(); j++) {
      double pd_weighted_input = next_cache->layer->weights_.ElementAt(i, j);
      pd_cost_weighted_input +=
        pd_weighted_input *
        next_cache->pd_cost_weighted_input->ElementAt(0, j);
    }
    cache->pd_cost_weighted_input->MutableElementAt(0, i) = pd_cost_weighted_input;
  }
  cache->pd_cost_weighted_input =
    cache->pd_cost_weighted_input->HadamardMult(GetActivationDeriv(activation_)(cache->w_input));
}

std::pair<Matrix, Matrix> Layer::FinishBackPropagate(LayerLearnCache* cache) const {
  ASSERT(cache->pd_cost_weighted_input.has_value());
  Matrix cost_gradient_weights = Matrix(weights_.RowCount(), weights_.ColCount());
  for (int32_t c = 0; c < cost_gradient_weights.ColCount(); c++) {
    for (int32_t r = 0; r < cost_gradient_weights.RowCount(); r++) {
      double pd_cost_weight =
        cache->input.ElementAt(0, r) *
        cache->pd_cost_weighted_input->ElementAt(0, c);
      cost_gradient_weights.MutableElementAt(r, c) = pd_cost_weight;
    }
  }

  Matrix cost_gradient_biases = Matrix(biases_.RowCount(), biases_.ColCount());
  for (int32_t c = 0; c < cost_gradient_biases.ColCount(); c++) {
    double pd_cost_bias = 1.0 * cache->pd_cost_weighted_input->ElementAt(0, c);
    cost_gradient_biases.MutableElementAt(0, c) = pd_cost_bias;
  }

  return std::make_pair(cost_gradient_weights, cost_gradient_biases);
}

void Layer::ApplyGradients(std::pair<Matrix, Matrix> gradients) {
  const Matrix cost_gradient_weights = std::move(gradients.first);
  double weight_decay = (1.0 - regularization_ * learn_rate_);
  weight_velocities_ =
    (weight_velocities_ * momentum_) -
    (cost_gradient_weights * learn_rate_);
  weights_ = weights_ * weight_decay + weight_velocities_;

  const Matrix cost_gradient_biases = std::move(gradients.second);
  bias_velocities_ =
    (bias_velocities_ * momentum_) -
    (cost_gradient_biases * learn_rate_);
  biases_ = biases_ + bias_velocities_;
}
