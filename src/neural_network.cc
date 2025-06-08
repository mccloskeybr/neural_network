#include "neural_network.h"

#include <iostream>
#include <ostream>
#include <vector>

#include "matrix.h"

NeuralNetwork::NeuralNetwork(
    Config cfg,
    std::vector<Matrix> weights,
    std::vector<Matrix> biases) {
  assert(weights.size() == biases.size());
  cfg_ = std::move(cfg);
  layers_.reserve(weights.size());
  for (int32_t i = 0; i < weights.size(); i++) {
    layers_.push_back(Layer(cfg_, std::move(weights[i]), std::move(biases[i])));
  }
}

NeuralNetwork NeuralNetwork::Random(
    Config cfg, std::vector<int32_t> layer_sizes) {
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
  return NeuralNetwork(std::move(cfg), std::move(weights), std::move(biases));
}

static float total_correct = 0.0f;
static float total = 0.0f;
float NeuralNetwork::Learn(Matrix input, int32_t expected_class) {
  std::vector<Layer::LearnCache> learn_caches(layers_.size());

  // feed forward
  Matrix layer_value = std::move(input);
  for (int32_t i = 0; i < layers_.size(); i++) {
    layer_value = layers_[i].FeedForward(layer_value, &learn_caches[i]);
  }

  // back propagate
  Matrix expected_output = Matrix(1, layers_[layers_.size() - 1].OutputSize());
  expected_output.ElementAt(0, expected_class) = 1.0f;
  layers_[layers_.size() - 1].CalcPDCostWeightedInputOutput(
      &learn_caches[learn_caches.size() - 1], expected_output);
  for (int32_t i = layers_.size() - 2; i >= 0; i--) {
    layers_[i].CalcPDCostWeightedInputIntermed(&learn_caches[i], &learn_caches[i + 1]);
  }
  for (int32_t i = 0; i < layers_.size(); i++) {
    layers_[i].FinishBackPropagate(&learn_caches[i]);
  }

  float cost = 0.0f;
  auto cost_fn = GetCost(cfg_.cost);
  for (int32_t i = 0; i < expected_output.ColCount(); i++) {
    cost += cost_fn(layer_value.ElementAt(0, i), expected_output.ElementAt(0, i));
  }


  int32_t actual = layer_value.Classify();
  int32_t expected = expected_class;
  total += 1;
  total_correct += (actual == expected);
  std::cout << actual << " : " << expected << " " << total_correct / total << std::endl;

  return cost;
}
