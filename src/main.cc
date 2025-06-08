#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include "assert.h"
#include "neural_network.h"
#include "matrix.h"

int main(int argc, char* argv[]) {
  auto neural_network = NeuralNetwork::Random(
      {.activation = Activation::SIGMOID,
       .cost = Cost::MEAN_SQUARED,
       .learn_rate = 0.05},
      {784, 128, 64, 10});

  std::ifstream file("Z:\\neural_network\\data\\mnist\\mnist_train.csv");
  if (!file.is_open()) {
    std::cerr << "Error opening file." << std::endl;
    exit(1);
  }

  std::string line;
  getline(file, line); // NOTE: eat headers
  while (getline(file, line)) {
    std::stringstream stream(line);

    uint32_t expected_class;
    std::vector<float> input_elements;
    input_elements.reserve(28 * 28);

    std::string field;
    std::getline(stream, field, ',');
    expected_class = std::stoul(field);

    while (std::getline(stream, field, ',')) {
      input_elements.push_back(std::stof(field) / 255.0f);
    }
    ASSERT(input_elements.size() == 28 * 28);

    Matrix input = Matrix(1, 28 * 28, input_elements);
    float cost = neural_network.Learn(input, expected_class);
    // std::cout << "Cost: " << cost << std::endl;
  }

  return 0;
}
