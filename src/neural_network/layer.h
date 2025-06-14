#ifndef SRC_LAYER_H_
#define SRC_LAYER_H_

#include <functional>
#include <optional>
#include <utility>

#include "absl/log/check.h"
#include "src/common/matrix.h"
#include "src/neural_network/activation.h"
#include "src/neural_network/cost.h"
#include "src/neural_network/params.h"
#include "src/protos/model_checkpoint.pb.h"

class Layer {
 public:
  explicit Layer(
      Matrix weights, Matrix biases,
      protos::Activation activation) :
    weights_(std::move(weights)),
    biases_(std::move(biases)),
    weight_velocities_(Matrix(weights_.RowCount(), weights_.ColCount())),
    bias_velocities_(Matrix(biases_.RowCount(), biases_.ColCount())),
    activation_(activation) {
      DCHECK(weights_.ColCount() == biases_.ColCount());
      DCHECK(biases_.RowCount() == 1);
    }

  int32_t InputSize() const;
  int32_t OutputSize() const;
  const Matrix& Weights() const;
  const Matrix& Biases() const;

  Matrix Infer(const Matrix& input) const;

  struct LayerLearnCache {
    const Layer* layer;
    Matrix input;
    Matrix w_input;
    Matrix activated;
    std::optional<Matrix> pd_cost_weighted_input;
  };
  Matrix FeedForward(const Matrix& input, LayerLearnCache* cache) const;
  void CalcPDCostWeightedInputOutput(
      const TrainParameters& train_params, LayerLearnCache* cache, const Matrix& expected_output) const;
  void CalcPDCostWeightedInputIntermed(LayerLearnCache* cache, LayerLearnCache* next_cache) const;
  std::pair<Matrix, Matrix> FinishBackPropagate(LayerLearnCache* cache) const;
  void ApplyGradients(const TrainParameters& train_params, std::pair<Matrix, Matrix> gradients);

 private:
  Matrix weights_;
  Matrix biases_;
  Matrix weight_velocities_;
  Matrix bias_velocities_;
  protos::Activation activation_;
};

#endif
