#ifndef SRC_NEURAL_NETWORK_H_
#define SRC_NEURAL_NETWORK_H_

#include <vector>

#include "src/common/matrix.h"
#include "src/neural_network/layer.h"
#include "src/neural_network/params.h"
#include "src/protos/model_checkpoint.pb.h"

class NeuralNetwork {
 public:
  static NeuralNetwork Random(
      const std::vector<int32_t> layer_sizes,
      protos::Activation intermed_activation,
      protos::Activation output_activation);
  static absl::StatusOr<NeuralNetwork> FromCheckpoint(
      const protos::ModelCheckpoint& checkpoint_proto);

  Matrix Infer(const Matrix& input) const;

  struct NetworkLearnCache {
    std::vector<Layer::LayerLearnCache> layer_caches;
  };
  Matrix FeedForward(
      const Matrix& input, NetworkLearnCache* cache) const;
  std::vector<std::pair<Matrix, Matrix>> BackPropagate(
      const TrainParameters& train_params, NetworkLearnCache* cache,
      const Matrix& actual_output, const Matrix& expected_output) const;
  void ApplyGradients(
      const TrainParameters& train_params,
      std::vector<std::pair<Matrix, Matrix>> gradients);

  int32_t LayersCount() const;
  const Layer& GetLayer(int32_t i) const;

  protos::ModelCheckpoint ToCheckpoint() const;

 protected:
  explicit NeuralNetwork(
      std::vector<Matrix> weights, std::vector<Matrix> biases,
      protos::Activation intermed_activation,
      protos::Activation output_activation);

 private:
  std::vector<Layer> layers_;
};

#endif
