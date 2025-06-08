#include "matrix.h"

#include <array>
#include <cassert>
#include <functional>
#include <iostream>
#include <ostream>
#include <random>
#include <string>
#include <utility>

Matrix Matrix::Random(int32_t row_count, int32_t col_count) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution rand;
  std::vector<float> result_elements(row_count * col_count);
  for (int32_t i = 0; i < result_elements.size(); i++) {
    result_elements[i] = rand(gen);
  }
  return Matrix(row_count, col_count, std::move(result_elements));
}

Matrix Matrix::Transpose() {
  std::vector<float> result_elements;
  result_elements.reserve(this->elements_.size());
  for (int32_t c = 0; c < this->col_count_; c++) {
    for (int32_t r = 0; r < this->row_count_; r++) {
      result_elements.push_back(this->ElementAt(r, c));
    }
  }
  return Matrix(col_count_, row_count_, std::move(result_elements));
}

Matrix Matrix::Map(std::function<float(float)> closure) {
  std::vector<float> elements_copy = this->elements_;
  for (float& x : elements_copy) { x = closure(x); }
  return Matrix(row_count_, col_count_, elements_copy);
}

Matrix Matrix::Merge(Matrix& other, std::function<float(float, float)> closure) {
  assert(this->row_count_ == other.row_count_);
  assert(this->col_count_ == other.col_count_);
  std::vector<float> result_elements;
  result_elements.reserve(this->elements_.size());
  for (int32_t i = 0; i < this->elements_.size(); i++) {
    result_elements.push_back(closure(this->elements_[i], other.elements_[i]));
  }
  return Matrix(row_count_, col_count_, result_elements);
}

Matrix Matrix::HadamardMult(Matrix& other) {
  assert(this->row_count_ == other.row_count_);
  assert(this->col_count_ == other.col_count_);
  std::vector<float> result_elements;
  result_elements.reserve(this->elements_.size());
  for (int32_t i = 0; i < this->elements_.size(); i++) {
    result_elements.push_back(this->elements_[i] * other.elements_[i]);
  }
  return Matrix(row_count_, col_count_, result_elements);
}

Matrix Matrix::operator*(float scalar) {
  Matrix result = *this;
  for (float& element : result.elements_) { element *= scalar; }
  return result;
}

Matrix Matrix::operator+(Matrix& other) {
  assert(this->row_count_ == other.row_count_);
  assert(this->col_count_ == other.col_count_);
  std::vector<float> result_elements(this->row_count_ * this->col_count_);
  for (int32_t i = 0; i < result_elements.size(); i++) {
    result_elements[i] = this->elements_[i] + other.elements_[i];
  }
  return Matrix(this->row_count_, this->col_count_, std::move(result_elements));
}

Matrix Matrix::operator*(Matrix& other) {
  assert(this->col_count_ == other.row_count_);
  Matrix result(this->row_count_, other.col_count_);
  for (int32_t i = 0; i < this->row_count_; i++) {
    for (int32_t k = 0; k < this->col_count_; k++) {
      for (int32_t j = 0; j < other.col_count_; j++) {
        result.ElementAt(i, j) +=
          this->ElementAt(i, k) * other.ElementAt(k, j);
      }
    }
  }
  return result;
}

bool Matrix::operator==(Matrix& other) {
  assert(this->row_count_ == other.row_count_);
  assert(this->col_count_ == other.col_count_);
  for (int32_t i = 0; i < this->elements_.size(); i++) {
    if (this->elements_[i] != other.elements_[i]) {
      return false;
    }
  }
  return true;
}

int32_t Matrix::Classify() {
  assert(row_count_ == 1);
  int32_t idx_max = 0;
  for (int32_t i = 1; i < elements_.size(); i++) {
    if (elements_[i] > elements_[idx_max]) {
      idx_max = i;
    }
  }
  return idx_max;
}

int32_t Matrix::RowCount() { return row_count_; }

int32_t Matrix::ColCount() { return col_count_; }

float& Matrix::ElementAt(int32_t r, int32_t c) {
  assert(r < row_count_ && c < col_count_);
  return this->elements_[(r * col_count_) + c];
}

std::string Matrix::DebugString() {
  std::string result;
  for (int32_t i = 0; i < this->row_count_; i++) {
    result += "| ";
    for (int32_t j = 0; j < this->col_count_; j++) {
      result += std::to_string(this->ElementAt(i, j));
      if (j < this->col_count_ - 1) { result += ", "; }
    }
    result += " |\n";
  }
  return result;
}
