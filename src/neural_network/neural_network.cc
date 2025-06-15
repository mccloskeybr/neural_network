#include "src/neural_network/neural_network.h"

#include <iostream>
#include <ostream>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/common/matrix.h"
#include "src/protos/model_checkpoint.pb.h"

NeuralNetwork::NeuralNetwork(
    std::vector<Matrix> weights, std::vector<Matrix> biases,
    protos::Activation intermed_activation,
    protos::Activation output_activation) {
  DCHECK(weights.size() == biases.size());
  layers_.reserve(weights.size());
  for (int32_t i = 0; i < weights.size(); i++) {
    auto activation = (i == (weights.size() - 1)) ? output_activation : intermed_activation;
    layers_.emplace_back(std::move(weights[i]), std::move(biases[i]), activation);
  }
}

NeuralNetwork NeuralNetwork::Random(
    const std::vector<int32_t> layer_sizes,
    protos::Activation intermed_activation,
    protos::Activation output_activation) {
  std::vector<Matrix> weights;
  std::vector<Matrix> biases;
  weights.reserve(layer_sizes.size() - 1);
  biases.reserve(layer_sizes.size() - 1);
  for (int32_t i = 0; i < layer_sizes.size() - 1; i++) {
    int32_t row_count = layer_sizes[i];
    int32_t col_count = layer_sizes[i + 1];
    weights.push_back(Matrix::Random(row_count, col_count));
    biases.push_back(Matrix::Random(1, col_count));
  }
  return NeuralNetwork(std::move(weights), std::move(biases), intermed_activation, output_activation);
}

absl::StatusOr<NeuralNetwork> NeuralNetwork::FromCheckpoint(const protos::ModelCheckpoint& checkpoint_proto) {
  std::vector<Matrix> weights;
  std::vector<Matrix> biases;
  weights.reserve(checkpoint_proto.layers().size() - 1);
  biases.reserve(checkpoint_proto.layers().size() - 1);

  for (int32_t i = 0; i < checkpoint_proto.layers().size(); i++) {
    const protos::Layer* curr_layer = &checkpoint_proto.layers()[i];
    const protos::Layer* next_layer = nullptr;
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
  return NeuralNetwork(std::move(weights), std::move(biases),
      checkpoint_proto.intermed_activation(), checkpoint_proto.output_activation());
}

protos::ModelCheckpoint NeuralNetwork::ToCheckpoint() const {
  protos::ModelCheckpoint checkpoint_proto;
  for (const Layer& layer : layers_) {
    protos::Layer& layer_proto = *checkpoint_proto.add_layers();;
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

Matrix NeuralNetwork::Infer(const Matrix& input) const {
  Matrix layer_value = input;
  for (int32_t i = 0; i < layers_.size(); i++) {
    layer_value = layers_[i].Infer(layer_value);
  }
  return layer_value;
}

Matrix NeuralNetwork::FeedForward(const Matrix& input, NetworkLearnCache* cache) const {
  *cache = NetworkLearnCache {
    .layer_caches = std::vector<Layer::LayerLearnCache>(layers_.size()),
  };
  Matrix layer_value = input;
  for (int32_t i = 0; i < layers_.size(); i++) {
    layer_value = layers_[i].FeedForward(layer_value, &cache->layer_caches[i]);
  }
  return layer_value;
}

std::vector<std::pair<Matrix, Matrix>> NeuralNetwork::BackPropagate(
    const TrainParameters& train_params, NetworkLearnCache* cache,
    const Matrix& actual_output, const Matrix& expected_output) const {
  DCHECK(cache != nullptr);
  std::vector<std::pair<Matrix, Matrix>> gradients(layers_.size());
  int32_t output_idx = layers_.size() - 1;
  layers_[output_idx].CalcPDCostWeightedInputOutput(
      train_params, &cache->layer_caches[output_idx], expected_output);
  gradients[output_idx] = layers_[output_idx].FinishBackPropagate(
      &cache->layer_caches[cache->layer_caches.size() - 1]);
  for (int32_t i = layers_.size() - 2; i >= 0; i--) {
    layers_[i].CalcPDCostWeightedInputIntermed(
        &cache->layer_caches[i], &cache->layer_caches[i + 1]);
    gradients[i] = layers_[i].FinishBackPropagate(&cache->layer_caches[i]);
  }
  return gradients;
}

void NeuralNetwork::ApplyGradients(
    const TrainParameters& train_params,
    std::vector<std::pair<Matrix, Matrix>> gradients) {
  DCHECK(gradients.size() == layers_.size());
  for (int32_t i = 0; i < gradients.size(); i++) {
    layers_[i].ApplyGradients(train_params, std::move(gradients[i]));
  }
}
