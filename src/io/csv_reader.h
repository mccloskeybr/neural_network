#ifndef SRC_IO_CSV_READER_H_
#define SRC_IO_CSV_READER_H_

#include <fstream>
#include <cstdint>
#include <iostream>
#include <optional>
#include <utility>

#include "src/common/matrix.h"

class CsvReader {
 public:
  static std::optional<CsvReader> Open(std::string filename);
  std::optional<std::pair<uint32_t, Matrix>> GetNextSample();

 protected:
  CsvReader(std::ifstream file) : file_(std::move(file)) {}

 private:
  std::ifstream file_;
};

#endif
