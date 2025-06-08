#include "layer.h"

#include <optional>

#include "assert.h"
#include "activation.h"
#include "config.h"
#include "cost.h"

int32_t Layer::InputSize() { return weights_.RowCount(); }

int32_t Layer::OutputSize() { return weights_.ColCount(); }

Matrix Layer::FeedForward(Matrix input, LearnCache* cache) {
  Matrix w_input = (input * weights_);
  Matrix activated = (w_input + biases_).Map(GetActivation(cfg_.activation));
  if (cache != nullptr) {
    *cache = LearnCache {
      .layer = this,
      .input = input,
      .w_input = w_input,
      .activated = activated,
      .pd_cost_weighted_input = std::nullopt,
    };
  }
  return activated;
}

void Layer::CalcPDCostWeightedInputOutput(LearnCache* cache, Matrix expected_output) {
  Matrix pd_cost_activation = cache->activated.Merge(expected_output, GetCostDeriv(cfg_.cost));
  Matrix pd_activation_weighted_input = cache->w_input.Map(GetActivationDeriv(cfg_.activation));
  cache->pd_cost_weighted_input = pd_cost_activation.HadamardMult(pd_activation_weighted_input);
}

void Layer::CalcPDCostWeightedInputIntermed(LearnCache* cache, LearnCache* next_cache) {
  ASSERT(next_cache->pd_cost_weighted_input.has_value());
  cache->pd_cost_weighted_input = Matrix(1, next_cache->layer->weights_.RowCount());
  for (int32_t i = 0; i < next_cache->layer->weights_.RowCount(); i++) {
    float pd_cost_weighted_input = 0.0f;
    for (int32_t j = 0; j < next_cache->layer->weights_.ColCount(); j++) {
      float pd_weighted_input = next_cache->layer->weights_.ElementAt(i, j);
      pd_cost_weighted_input +=
        pd_weighted_input *
        next_cache->pd_cost_weighted_input->ElementAt(0, j);
    }
    cache->pd_cost_weighted_input->ElementAt(0, i) = pd_cost_weighted_input;
  }
}

void Layer::FinishBackPropagate(LearnCache* cache) {
  ASSERT(cache->pd_cost_weighted_input.has_value());
  Matrix cost_gradient_weights = Matrix(weights_.RowCount(), weights_.ColCount());
  for (int32_t c = 0; c < cost_gradient_weights.ColCount(); c++) {
    for (int32_t r = 0; r < cost_gradient_weights.RowCount(); r++) {
      float pd_cost_weight =
        cache->input.ElementAt(0, r) *
        cache->pd_cost_weighted_input->ElementAt(0, c);
      cost_gradient_weights.ElementAt(r, c) = pd_cost_weight;
    }
  }

  Matrix cost_gradient_biases = Matrix(biases_.RowCount(), biases_.ColCount());
  for (int32_t c = 0; c < cost_gradient_biases.ColCount(); c++) {
    float pd_cost_bias = 1.0f * cache->pd_cost_weighted_input->ElementAt(0, c);
    cost_gradient_biases.ElementAt(0, c) = pd_cost_bias;
  }

  weights_ = weights_.Merge(
      cost_gradient_weights,
      [&](float weight, float cost_gradient) {
        return weight + -1.0f * cost_gradient * this->cfg_.learn_rate;
      });
  biases_ = biases_.Merge(
      cost_gradient_biases,
      [&](float bias, float cost_gradient) {
        return bias + -1.0f * cost_gradient * this->cfg_.learn_rate;
      });
}
