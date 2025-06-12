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
    .layer_sizes = {784, 128, 64, 10},
    .activation = Activation::SIGMOID,
    .cost = Cost::MEAN_SQUARED,
    .learn_rate = 0.005,
    .num_threads = std::thread::hardware_concurrency(),
    .batch_size = 3,
    .num_epochs = 1,
  };
  NeuralNetwork neural_network = Train(std::move(params), std::move(*reader));

  return 0;
}
