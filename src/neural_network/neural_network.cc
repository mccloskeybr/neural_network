#include "src/neural_network/neural_network.h"

#include <iostream>
#include <ostream>
#include <vector>

#include "src/common/assert.h"
#include "src/common/matrix.h"

NeuralNetwork::NeuralNetwork(
    Parameters params,
    std::vector<Matrix> weights,
    std::vector<Matrix> biases) {
  ASSERT(weights.size() == biases.size());
  params_ = std::move(params);
  layers_.reserve(weights.size());
  for (int32_t i = 0; i < weights.size(); i++) {
    Activation activation = (i == (weights.size() - 1)) ?
      params_.output_activation : params_.intermed_activation;
    layers_.emplace_back(
        std::move(weights[i]), std::move(biases[i]),
        activation, params_.cost,
        params_.learn_rate, params.momentum, params.regularization);
  }
}

NeuralNetwork NeuralNetwork::Random(Parameters params) {
  std::vector<Matrix> weights;
  std::vector<Matrix> biases;
  weights.reserve(params.layer_sizes.size() - 1);
  biases.reserve(params.layer_sizes.size() - 1);
  for (int32_t i = 0; i < params.layer_sizes.size() - 1; i++) {
    int32_t row_count = params.layer_sizes[i];
    int32_t col_count = params.layer_sizes[i + 1];
    weights.push_back(Matrix::Random(row_count, col_count));
    biases.push_back(Matrix::Random(1, col_count));
  }
  return NeuralNetwork(std::move(params), std::move(weights), std::move(biases));
}

int32_t NeuralNetwork::LayersCount() const {
  return layers_.size();
}

const Layer& NeuralNetwork::GetLayer(int32_t i) const {
  return layers_[i];
}

Matrix NeuralNetwork::FeedForward(Matrix input, NetworkLearnCache* cache) const {
  if (cache != nullptr) {
    *cache = NetworkLearnCache {
      .layer_caches = std::vector<Layer::LayerLearnCache>(layers_.size()),
    };
  }
  Matrix layer_value = std::move(input);


  for (int32_t i = 0; i < layers_.size(); i++) {
    layer_value = layers_[i].FeedForward(
        layer_value, cache != nullptr ? &cache->layer_caches[i] : nullptr);
  }
  return layer_value;
}

std::vector<std::pair<Matrix, Matrix>> NeuralNetwork::BackPropagate(
    Matrix actual_output, Matrix expected_output, NetworkLearnCache* cache) const {
  ASSERT(cache != nullptr);

  layers_[layers_.size() - 1].CalcPDCostWeightedInputOutput(
      &cache->layer_caches[cache->layer_caches.size() - 1], expected_output);
  for (int32_t i = layers_.size() - 2; i >= 0; i--) {
    layers_[i].CalcPDCostWeightedInputIntermed(
        &cache->layer_caches[i], &cache->layer_caches[i + 1]);
  }
  std::vector<std::pair<Matrix, Matrix>> gradients(layers_.size());
  for (int32_t i = 0; i < layers_.size(); i++) {
    gradients[i] = layers_[i].FinishBackPropagate(&cache->layer_caches[i]);
  }

  return gradients;
}

void NeuralNetwork::ApplyGradients(std::vector<std::pair<Matrix, Matrix>> gradients) {
  ASSERT(gradients.size() == layers_.size());
  for (int32_t i = 0; i < gradients.size(); i++) {
    layers_[i].ApplyGradients(std::move(gradients[i]));
  }
}
