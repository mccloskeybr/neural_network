#include "src/io/csv_reader.h"

#include <fstream>
#include <iostream>
#include <ostream>
#include <optional>
#include <sstream>

#include "src/common/matrix.h"

std::optional<CsvReader> CsvReader::Open(std::string filename) {
  std::ifstream file(filename);
  if (!file.is_open()) { return std::nullopt; }
  std::string line;
  getline(file, line); // NOTE: eat headers
  return CsvReader(std::move(file));
}

std::optional<std::pair<uint32_t, Matrix>> CsvReader::GetNextSample() {
  std::string line;
  getline(file_, line);
  std::stringstream stream(line);
  std::string field;

  uint32_t expected_class;
  std::getline(stream, field, ',');
  expected_class = std::stoul(field);

  std::vector<float> input_elements;
  while (std::getline(stream, field, ',')) {
    input_elements.push_back(std::stof(field) / 255.0f);
  }
  Matrix input = Matrix(1, input_elements.size(), input_elements);

  return std::make_pair(expected_class, input);
}
