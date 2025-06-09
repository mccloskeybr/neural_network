#include "src/neural_network/layer.h"

#include <optional>

#include "src/params.h"
#include "src/common/assert.h"
#include "src/neural_network/activation.h"
#include "src/neural_network/cost.h"

int32_t Layer::InputSize() const { return weights_.RowCount(); }

int32_t Layer::OutputSize() const { return weights_.ColCount(); }

Matrix Layer::FeedForward(const Matrix input, LayerLearnCache* cache) const {
  const Matrix w_input = (input * weights_);
  const Matrix activated = (w_input + biases_).Map(GetActivation(params_->activation));
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
  const Matrix pd_cost_activation = cache->activated.Merge(expected_output, GetCostDeriv(params_->cost));
  const Matrix pd_activation_weighted_input = cache->w_input.Map(GetActivationDeriv(params_->activation));
  cache->pd_cost_weighted_input = pd_cost_activation.HadamardMult(pd_activation_weighted_input);
}

void Layer::CalcPDCostWeightedInputIntermed(LayerLearnCache* cache, LayerLearnCache* next_cache) const {
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
    cache->pd_cost_weighted_input->MutableElementAt(0, i) = pd_cost_weighted_input;
  }
}

std::pair<Matrix, Matrix> Layer::FinishBackPropagate(LayerLearnCache* cache) const {
  ASSERT(cache->pd_cost_weighted_input.has_value());
  Matrix cost_gradient_weights = Matrix(weights_.RowCount(), weights_.ColCount());
  for (int32_t c = 0; c < cost_gradient_weights.ColCount(); c++) {
    for (int32_t r = 0; r < cost_gradient_weights.RowCount(); r++) {
      float pd_cost_weight =
        cache->input.ElementAt(0, r) *
        cache->pd_cost_weighted_input->ElementAt(0, c);
      cost_gradient_weights.MutableElementAt(r, c) = pd_cost_weight;
    }
  }

  Matrix cost_gradient_biases = Matrix(biases_.RowCount(), biases_.ColCount());
  for (int32_t c = 0; c < cost_gradient_biases.ColCount(); c++) {
    float pd_cost_bias = 1.0f * cache->pd_cost_weighted_input->ElementAt(0, c);
    cost_gradient_biases.MutableElementAt(0, c) = pd_cost_bias;
  }

  return std::make_pair(cost_gradient_weights, cost_gradient_biases);
}

void Layer::ApplyGradients(std::pair<Matrix, Matrix> gradients) {
  const Matrix cost_gradient_weights = std::move(gradients.first);
  weights_ = weights_.Merge(
      cost_gradient_weights,
      [&](float weight, float cost_gradient) {
        return weight + -1.0f * cost_gradient * params_->learn_rate;
      });

  const Matrix cost_gradient_biases = std::move(gradients.second);
  biases_ = biases_.Merge(
      cost_gradient_biases,
      [&](float bias, float cost_gradient) {
        return bias + -1.0f * cost_gradient * params_->learn_rate;
      });
}
