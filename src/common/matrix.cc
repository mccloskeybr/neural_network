#include "src/common/matrix.h"

#include <array>
#include <functional>
#include <iostream>
#include <ostream>
#include <random>
#include <string>
#include <utility>

#include "src/common/assert.h"

Matrix Matrix::Random(int32_t row_count, int32_t col_count) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution rand;
  std::vector<double> result_elements(row_count * col_count);
  for (int32_t i = 0; i < result_elements.size(); i++) {
    result_elements[i] = rand(gen);
  }
  return Matrix(row_count, col_count, std::move(result_elements));
}

Matrix Matrix::Transpose() const {
  std::vector<double> result_elements;
  result_elements.reserve(elements_.size());
  for (int32_t c = 0; c < col_count_; c++) {
    for (int32_t r = 0; r < row_count_; r++) {
      result_elements.push_back(ElementAt(r, c));
    }
  }
  return Matrix(col_count_, row_count_, std::move(result_elements));
}

Matrix Matrix::Map(std::function<double(double)> closure) const {
  std::vector<double> elements_copy = elements_;
  for (double& x : elements_copy) { x = closure(x); }
  return Matrix(row_count_, col_count_, elements_copy);
}

Matrix Matrix::Merge(const Matrix& other, std::function<double(double, double)> closure) const {
  ASSERT(row_count_ == other.row_count_);
  ASSERT(col_count_ == other.col_count_);
  std::vector<double> result_elements;
  result_elements.reserve(elements_.size());
  for (int32_t i = 0; i < elements_.size(); i++) {
    result_elements.push_back(closure(elements_[i], other.elements_[i]));
  }
  return Matrix(row_count_, col_count_, result_elements);
}

Matrix Matrix::HadamardMult(const Matrix& other) const {
  ASSERT(row_count_ == other.row_count_);
  ASSERT(col_count_ == other.col_count_);
  std::vector<double> result_elements;
  result_elements.reserve(elements_.size());
  for (int32_t i = 0; i < elements_.size(); i++) {
    result_elements.push_back(elements_[i] * other.elements_[i]);
  }
  return Matrix(row_count_, col_count_, result_elements);
}

Matrix Matrix::operator*(double scalar) const {
  Matrix result = *this;
  for (double& element : result.elements_) { element *= scalar; }
  return result;
}

Matrix Matrix::operator+(const Matrix& other) const {
  ASSERT(row_count_ == other.row_count_);
  ASSERT(col_count_ == other.col_count_);
  std::vector<double> result_elements(row_count_ * col_count_);
  for (int32_t i = 0; i < result_elements.size(); i++) {
    result_elements[i] = elements_[i] + other.elements_[i];
  }
  return Matrix(row_count_, col_count_, std::move(result_elements));
}

Matrix Matrix::operator-(const Matrix& other) const {
  ASSERT(row_count_ == other.row_count_);
  ASSERT(col_count_ == other.col_count_);
  std::vector<double> result_elements(row_count_ * col_count_);
  for (int32_t i = 0; i < result_elements.size(); i++) {
    result_elements[i] = elements_[i] - other.elements_[i];
  }
  return Matrix(row_count_, col_count_, std::move(result_elements));
}

Matrix Matrix::operator*(const Matrix& other) const {
  ASSERT(col_count_ == other.row_count_);
  Matrix result(row_count_, other.col_count_);
  for (int32_t i = 0; i < row_count_; i++) {
    for (int32_t k = 0; k < col_count_; k++) {
      for (int32_t j = 0; j < other.col_count_; j++) {
        result.MutableElementAt(i, j) += ElementAt(i, k) * other.ElementAt(k, j);
      }
    }
  }
  return result;
}

bool Matrix::operator==(const Matrix& other) const {
  ASSERT(row_count_ == other.row_count_);
  ASSERT(col_count_ == other.col_count_);
  for (int32_t i = 0; i < elements_.size(); i++) {
    if (elements_[i] != other.elements_[i]) {
      return false;
    }
  }
  return true;
}

int32_t Matrix::Classify() const {
  ASSERT(row_count_ == 1);
  int32_t idx_max = 0;
  for (int32_t i = 1; i < elements_.size(); i++) {
    if (elements_[i] > elements_[idx_max]) {
      idx_max = i;
    }
  }
  return idx_max;
}

int32_t Matrix::RowCount() const { return row_count_; }

int32_t Matrix::ColCount() const { return col_count_; }

double Matrix::ElementAt(int32_t r, int32_t c) const {
  ASSERT(r < row_count_ && c < col_count_);
  return elements_[(r * col_count_) + c];
}

double& Matrix::MutableElementAt(int32_t r, int32_t c) {
  ASSERT(r < row_count_ && c < col_count_);
  return elements_[(r * col_count_) + c];
}

std::string Matrix::DebugString() const {
  std::string result;
  for (int32_t i = 0; i < row_count_; i++) {
    result += "| ";
    for (int32_t j = 0; j < col_count_; j++) {
      result += std::to_string(ElementAt(i, j));
      if (j < col_count_ - 1) { result += ", "; }
    }
    result += " |\n";
  }
  return result;
}
