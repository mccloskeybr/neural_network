#include "src/neural_network/cost.h"

#include <cmath>
#include <functional>

#include "src/common/assert.h"

double MeanSquaredError(double actual, double expected) {
  double error = actual - expected;
  return 0.5f * error * error;
}

double MeanSquaredErrorDeriv(double actual, double expected) {
  return actual - expected;
}

std::function<double(double, double)> GetCost(Cost cost) {
  using enum Cost;
  switch (cost) {
    case MEAN_SQUARED: { return MeanSquaredError; }
    default: { UNREACHABLE(); return MeanSquaredError; }
  }
}

std::function<double(double, double)> GetCostDeriv(Cost cost) {
  using enum Cost;
  switch (cost) {
    case MEAN_SQUARED: { return MeanSquaredErrorDeriv; }
    default: { UNREACHABLE(); return MeanSquaredErrorDeriv; }
  }
}

