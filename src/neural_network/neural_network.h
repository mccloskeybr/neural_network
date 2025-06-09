#ifndef SRC_NEURAL_NETWORK_H_
#define SRC_NEURAL_NETWORK_H_

#include <vector>

#include "src/params.h"
#include "src/common/matrix.h"
#include "src/neural_network/layer.h"

class NeuralNetwork {
 public:
  explicit NeuralNetwork(
      Parameters params, std::vector<Matrix> weights, std::vector<Matrix> biases);
  static NeuralNetwork Random(Parameters params);

  struct NetworkLearnCache {
    std::vector<Layer::LayerLearnCache> layer_caches;
  };

  Matrix FeedForward(
      Matrix input, NetworkLearnCache* cache) const;
  std::vector<std::pair<Matrix, Matrix>> BackPropagate(
      Matrix actual_output, Matrix expected_output, NetworkLearnCache* cache) const;
  void ApplyGradients(std::vector<std::pair<Matrix, Matrix>> gradients);

  int32_t LayersCount() const;
  const Layer& GetLayer(int32_t i) const;

 private:
  Parameters params_;
  std::vector<Layer> layers_;
};

#endif
