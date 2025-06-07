#ifndef SRC_MATRIX_H_
#define SRC_MATRIX_H_

#include <array>
#include <utility>

template<size_t R, size_t C>
struct Matrix {
  explicit Matrix(std::array<std::array<float, C>, R> elements) :
    e(std::move(elements)) {}
  explicit Matrix() {
    for (size_t i = 0; i < R; i++) {
      for (size_t j = 0; j < C; j++) {
        this->e[i][j] = 0;
      }
    }
  }

  Matrix<R, C> operator+(const Matrix<R, C>& other) {
    Matrix<R, C> result;
    for (size_t i = 0; i < R; i++) {
      for (size_t j = 0; j < C; j++) {
        result.e[i][j] = e[i][j] + other.e[i][j];
      }
    }
    return result;
  }

  template<size_t Co>
  Matrix<R, Co> operator*(const Matrix<C, Co>& other) {
    Matrix<R, Co> result;
    for (size_t i = 0; i < R; i++) {
      for (size_t k = 0; k < C; k++) {
        for (size_t j = 0; j < Co; j++) {
          result.e[i][j] += e[i][k] * other.e[k][j];
        }
      }
    }
    return result;
  }

  bool operator==(const Matrix<R, C>& other) {
    for (size_t i = 0; i < R; i++) {
      for (size_t j = 0; j < C; j++) {
        if (e[i][j] != other.e[i][j]) {
          return false;
        }
      }
    }
    return true;
  }

  void DebugPrint() {
    for (size_t i = 0; i < R; i++) {
      for (size_t j = 0; j < C; j++) {
        std::cout << e[i][j] << " ";
      }
      std::cout << std::endl;
    }
  }

  std::array<std::array<float, C>, R> e;
};

#endif
