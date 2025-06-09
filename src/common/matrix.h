#ifndef SRC_COMMON_MATRIX_H_
#define SRC_COMMON_MATRIX_H_

#include <array>
#include <functional>
#include <random>
#include <utility>

#include "src/common/assert.h"

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

  Matrix Transpose() const;
  Matrix Map(std::function<float(float)> closure) const;
  Matrix Merge(const Matrix& other, std::function<float(float, float)> closure) const;
  Matrix HadamardMult(const Matrix& other) const;
  Matrix operator+(const Matrix& other) const;
  Matrix operator*(float scalar) const;
  Matrix operator*(const Matrix& other) const;
  bool operator==(const Matrix& other) const;

  int32_t Classify() const;
  int32_t RowCount() const;
  int32_t ColCount() const;
  float ElementAt(int32_t r, int32_t c) const;
  float& MutableElementAt(int32_t r, int32_t c);
  std::string DebugString() const;

 private:
  int32_t row_count_;
  int32_t col_count_;
  std::vector<float> elements_;
};

#endif
