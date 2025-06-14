#include "src/neural_network/cost.h"

#include <cmath>
#include <functional>

#include "absl/log/check.h"
#include "src/common/matrix.h"

Matrix MeanSquaredError(const Matrix& actual, const Matrix& expected) {
  return (actual - expected) * 0.5;
}

Matrix MeanSquaredErrorDeriv(const Matrix& actual, const Matrix& expected) {
  return (actual - expected);
}

std::function<Matrix(const Matrix&, const Matrix&)> GetCost(Cost cost) {
  using enum Cost;
  switch (cost) {
    case MEAN_SQUARED: { return MeanSquaredError; }
    default: { CHECK(false); return MeanSquaredError; }
  }
}

std::function<Matrix(const Matrix&, const Matrix&)> GetCostDeriv(Cost cost) {
  using enum Cost;
  switch (cost) {
    case MEAN_SQUARED: { return MeanSquaredErrorDeriv; }
    default: { CHECK(false); return MeanSquaredErrorDeriv; }
  }
}

