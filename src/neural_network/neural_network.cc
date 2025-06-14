#include "src/neural_network/neural_network.h"

#include <iostream>
#include <ostream>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/common/matrix.h"

NeuralNetwork::NeuralNetwork(
    Parameters params,
    std::vector<Matrix> weights,
    std::vector<Matrix> biases) {
  DCHECK(weights.size() == biases.size());
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

absl::StatusOr<NeuralNetwork> NeuralNetwork::FromCheckpoint(
    const protos::ModelCheckpoint& checkpoint_proto, Parameters params) {
  std::vector<Matrix> weights;
  std::vector<Matrix> biases;
  weights.reserve(checkpoint_proto.layers().size() - 1);
  biases.reserve(checkpoint_proto.layers().size() - 1);

  for (int32_t i = 0; i < checkpoint_proto.layers().size(); i++) {
    const protos::ModelCheckpoint::Layer* curr_layer = &checkpoint_proto.layers()[i];
    const protos::ModelCheckpoint::Layer* next_layer = nullptr;
    if (i < checkpoint_proto.layers().size() - 1) {
      next_layer = &checkpoint_proto.layers()[i + 1];
    }
    if (next_layer != nullptr) {
      if (curr_layer->col_count() != next_layer->row_count()) {
        return absl::InvalidArgumentError(absl::StrCat(
              "Invalid weight dimensions! Layer: ", i,
              " has column count: ", curr_layer->col_count(),
              ", while layer: ", i + 1,
              " has row count: ", next_layer->row_count()));
      }
    }
    weights.emplace_back(Matrix(
        curr_layer->row_count(), curr_layer->col_count(),
        {curr_layer->weights().begin(), curr_layer->weights().end()}));
    biases.emplace_back(Matrix(
        1, curr_layer->col_count(),
        {curr_layer->biases().begin(), curr_layer->biases().end()}));
  }
  return NeuralNetwork(std::move(params), std::move(weights), std::move(biases));
}

protos::ModelCheckpoint NeuralNetwork::ToCheckpoint() const {
  protos::ModelCheckpoint checkpoint_proto;
  for (const Layer& layer : layers_) {
    protos::ModelCheckpoint::Layer& layer_proto = *checkpoint_proto.add_layers();;
    layer_proto.set_row_count(layer.Weights().RowCount());
    layer_proto.set_col_count(layer.Weights().ColCount());
    *layer_proto.mutable_weights() =
      {layer.Weights().Elements().begin(), layer.Weights().Elements().end()};
    *layer_proto.mutable_biases() =
      {layer.Biases().Elements().begin(), layer.Biases().Elements().end()};
  }
  return checkpoint_proto;
}

int32_t NeuralNetwork::LayersCount() const { return layers_.size(); }

const Layer& NeuralNetwork::GetLayer(int32_t i) const { return layers_[i]; }

Matrix NeuralNetwork::FeedForward(const Matrix& input, NetworkLearnCache* cache) const {
  if (cache != nullptr) {
    *cache = NetworkLearnCache {
      .layer_caches = std::vector<Layer::LayerLearnCache>(layers_.size()),
    };
  }
  Matrix layer_value = input;
  for (int32_t i = 0; i < layers_.size(); i++) {
    layer_value = layers_[i].FeedForward(
        layer_value, cache != nullptr ? &cache->layer_caches[i] : nullptr);
  }
  return layer_value;
}

std::vector<std::pair<Matrix, Matrix>> NeuralNetwork::BackPropagate(
    const Matrix& actual_output, const Matrix& expected_output, NetworkLearnCache* cache) const {
  DCHECK(cache != nullptr);
  std::vector<std::pair<Matrix, Matrix>> gradients(layers_.size());

  int32_t output_idx = layers_.size() - 1;
  layers_[output_idx].CalcPDCostWeightedInputOutput(
      &cache->layer_caches[output_idx], expected_output);
  gradients[output_idx] = layers_[output_idx].FinishBackPropagate(
      &cache->layer_caches[cache->layer_caches.size() - 1]);
  for (int32_t i = layers_.size() - 2; i >= 0; i--) {
    layers_[i].CalcPDCostWeightedInputIntermed(
        &cache->layer_caches[i], &cache->layer_caches[i + 1]);
    gradients[i] = layers_[i].FinishBackPropagate(&cache->layer_caches[i]);
  }

  return gradients;
}

void NeuralNetwork::ApplyGradients(const std::vector<std::pair<Matrix, Matrix>>& gradients) {
  DCHECK(gradients.size() == layers_.size());
  for (int32_t i = 0; i < gradients.size(); i++) {
    layers_[i].ApplyGradients(gradients[i]);
  }
}
