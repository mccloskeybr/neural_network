#ifndef SRC_MATRIX_H_
#define SRC_MATRIX_H_

#include <array>
#include <functional>
#include <random>
#include <utility>

#include "assert.h"

class Matrix {
 public:
  Matrix() : Matrix(0, 0) {}
  explicit Matrix(int32_t row_count, int32_t col_count) :
    row_count_(row_count),
    col_count_(col_count),
    elements_(std::vector<float>(row_count_ * col_count_)) {};
  explicit Matrix(int32_t row_count, int32_t col_count, std::vector<float> elements) :
    row_count_(row_count),
    col_count_(col_count),
    elements_(std::move(elements)) {
      ASSERT(elements_.size() > 0);
      ASSERT(elements_.size() == row_count_ * col_count_);
    };
  static Matrix Random(int32_t row_count, int32_t col_count);

  Matrix Transpose();
  Matrix Map(std::function<float(float)> closure);
  Matrix Merge(Matrix& other, std::function<float(float, float)> closure);
  Matrix HadamardMult(Matrix& other);
  Matrix operator+(Matrix& other);
  Matrix operator*(float scalar);
  Matrix operator*(Matrix& other);
  bool operator==(Matrix& other);

  int32_t Classify();
  int32_t RowCount();
  int32_t ColCount();
  float& ElementAt(int32_t r, int32_t c);
  std::string DebugString();

 private:
  int32_t row_count_;
  int32_t col_count_;
  std::vector<float> elements_;
};

#endif
