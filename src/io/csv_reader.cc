#include "src/io/csv_reader.h"

#include <fstream>
#include <iostream>
#include <ostream>
#include <optional>
#include <sstream>
#include <vector>

#include "src/common/matrix.h"

std::optional<CsvReader> CsvReader::Open(std::string filename) {
  std::ifstream file(filename);
  if (!file.is_open()) { return std::nullopt; }
  std::string line;
  getline(file, line); // NOTE: eat headers
  return CsvReader(std::move(file));
}

void CsvReader::Reset() {
  file_.clear();
  file_.seekg(0, std::ios::beg);
  std::string line;
  getline(file_, line); // NOTE: eat headers
}

std::optional<std::pair<uint32_t, Matrix>>
CsvReader::GetNextSample() {
  std::string line;
  if (!getline(file_, line)) {
    return std::nullopt;
  }
  std::stringstream stream(line);
  std::string field;

  uint32_t expected_class;
  std::getline(stream, field, ',');
  expected_class = std::stoul(field);

  std::vector<float> input_elements;
  while (std::getline(stream, field, ',')) {
    input_elements.push_back(std::stof(field));
  }
  Matrix input = Matrix(1, input_elements.size(), input_elements);

  return std::make_optional(std::make_pair(expected_class, input));
}

std::vector<std::pair<uint32_t, Matrix>>
CsvReader::GetNextBatchSample(int32_t batch_size) {
  std::vector<std::pair<uint32_t, Matrix>> batch;
  batch.reserve(batch_size);
  for (int32_t i = 0; i < batch_size; i++) {
    std::optional<std::pair<uint32_t, Matrix>> sample = GetNextSample();
    if (!sample.has_value()) { break; }
    batch.emplace_back(*sample);
  }
  return batch;
}
