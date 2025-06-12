#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <utility>

#include "src/common/assert.h"
#include "src/common/matrix.h"
#include "src/io/csv_reader.h"
#include "src/io/model_checkpoint.h"
#include "src/neural_network/neural_network.h"
#include "src/neural_network/trainer.h"

int main(int argc, char* argv[]) {
  std::optional<CsvReader> reader = CsvReader::Open(
      "Z:\\neural_network\\data\\mnist\\mnist_train.csv");
  if (!reader.has_value()) {
    std::cerr << "Error opening file." << std::endl;
    exit(1);
  }
  Parameters params = {
    .layer_sizes = {784, 200, 80, 10},
    .intermed_activation = Activation::SIGMOID,
    .output_activation = Activation::SOFTMAX,
    .cost = Cost::MEAN_SQUARED,
    .learn_rate = 0.05,
    .momentum = 0.5,
    .regularization = 0.0,
    .num_threads = std::thread::hardware_concurrency(),
    .batch_size = 10,
    .num_epochs = 1,
  };
  NeuralNetwork neural_network = Train(std::move(params), std::move(*reader));
  WriteModelCheckpoint(
      "Z:\\neural_network\\data\\checkpoints\\mnist.model_checkpoint.pb",
      neural_network.ToCheckpoint());

  return 0;
}
