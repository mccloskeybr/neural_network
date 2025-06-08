#include "cost.h"

#include <cassert>
#include <cmath>
#include <functional>

float MeanSquaredError(float actual, float expected) {
  float error = actual - expected;
  return 0.5f * error * error;
}

float MeanSquaredErrorDeriv(float actual, float expected) {
  return actual - expected;
}

std::function<float(float, float)> GetCost(Cost cost) {
  using enum Cost;
  switch (cost) {
    case MEAN_SQUARED: { return MeanSquaredError; }
    default: { assert(false); return MeanSquaredError; }
  }
}

std::function<float(float, float)> GetCostDeriv(Cost cost) {
  using enum Cost;
  switch (cost) {
    case MEAN_SQUARED: { return MeanSquaredErrorDeriv; }
    default: { assert(false); return MeanSquaredErrorDeriv; }
  }
}

