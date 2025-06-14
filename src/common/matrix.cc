#include "src/common/matrix.h"

#include <array>
#include <functional>
#include <iostream>
#include <ostream>
#include <random>
#include <string>
#include <utility>

#include "absl/log/check.h"

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
  Matrix result = Matrix(col_count_, row_count_);
  for (int32_t c = 0; c < col_count_; c++) {
    for (int32_t r = 0; r < row_count_; r++) {
      result.MutableElementAt(c, r) = ElementAt(r, c);
    }
  }
  return result;
}

Matrix Matrix::HadamardMult(const Matrix& other) const {
  DCHECK(row_count_ == other.row_count_);
  DCHECK(col_count_ == other.col_count_);
  std::vector<double> result_elements;
  result_elements.reserve(elements_.size());
  for (int32_t i = 0; i < elements_.size(); i++) {
    result_elements.push_back(elements_[i] * other.elements_[i]);
  }
  return Matrix(row_count_, col_count_, result_elements);
}

void Matrix::HadamardMultInPlace(const Matrix& other) {
  DCHECK(row_count_ == other.row_count_);
  DCHECK(col_count_ == other.col_count_);
  for (int32_t i = 0; i < elements_.size(); i++) {
    elements_[i] *= other.elements_[i];
  }
}

Matrix Matrix::operator*(double scalar) const {
  Matrix result = *this;
  for (double& element : result.elements_) { element *= scalar; }
  return result;
}

void Matrix::operator*=(double scalar) {
  for (double& element : elements_) { element *= scalar; }
}

Matrix Matrix::operator+(const Matrix& other) const {
  DCHECK(row_count_ == other.row_count_);
  DCHECK(col_count_ == other.col_count_);
  std::vector<double> result_elements(row_count_ * col_count_);
  for (int32_t i = 0; i < result_elements.size(); i++) {
    result_elements[i] = elements_[i] + other.elements_[i];
  }
  return Matrix(row_count_, col_count_, std::move(result_elements));
}

void Matrix::operator+=(const Matrix& other) {
  DCHECK(row_count_ == other.row_count_);
  DCHECK(col_count_ == other.col_count_);
  for (int32_t i = 0; i < elements_.size(); i++) {
    elements_[i] += other.elements_[i];
  }
}

Matrix Matrix::operator-(const Matrix& other) const {
  DCHECK(row_count_ == other.row_count_);
  DCHECK(col_count_ == other.col_count_);
  std::vector<double> result_elements(row_count_ * col_count_);
  for (int32_t i = 0; i < result_elements.size(); i++) {
    result_elements[i] = elements_[i] - other.elements_[i];
  }
  return Matrix(row_count_, col_count_, std::move(result_elements));
}

void Matrix::operator-=(const Matrix& other) {
  DCHECK(row_count_ == other.row_count_);
  DCHECK(col_count_ == other.col_count_);
  for (int32_t i = 0; i < elements_.size(); i++) {
    elements_[i] -= other.elements_[i];
  }
}

Matrix Matrix::operator*(const Matrix& other) const {
  DCHECK(col_count_ == other.row_count_);
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
  DCHECK(row_count_ == other.row_count_);
  DCHECK(col_count_ == other.col_count_);
  for (int32_t i = 0; i < elements_.size(); i++) {
    if (elements_[i] != other.elements_[i]) {
      return false;
    }
  }
  return true;
}

int32_t Matrix::Classify() const {
  DCHECK(row_count_ == 1);
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
  DCHECK(r < row_count_ && c < col_count_);
  return elements_[(r * col_count_) + c];
}

double& Matrix::MutableElementAt(int32_t r, int32_t c) {
  DCHECK(r < row_count_ && c < col_count_);
  return elements_[(r * col_count_) + c];
}

const std::vector<double>& Matrix::Elements() const {
  return elements_;
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
