#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include "src/common/assert.h"
#include "src/common/matrix.h"
#include "src/io/csv_reader.h"
#include "src/neural_network/neural_network.h"

int main(int argc, char* argv[]) {
  auto neural_network = NeuralNetwork::Random(
      {.activation = Activation::SIGMOID,
       .cost = Cost::MEAN_SQUARED,
       .learn_rate = 0.05},
      {784, 128, 64, 10});

  std::optional<CsvReader> reader = CsvReader::Open(
      "Z:\\neural_network\\data\\mnist\\mnist_train.csv");
  if (!reader.has_value()) {
    std::cerr << "Error opening file." << std::endl;
    exit(1);
  }

  std::optional<std::pair<uint32_t, Matrix>> sample = reader->GetNextSample();
  while (sample.has_value()) {
    uint32_t expected_class = sample->first;
    Matrix input = sample->second;
    ASSERT(input.RowCount() == 1 && input.ColCount == 28 * 28);

    neural_network.Learn(std::move(input), expected_class);

    sample = reader->GetNextSample();
  }

  return 0;
}
