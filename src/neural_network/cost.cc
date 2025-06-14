#include "src/neural_network/cost.h"

#include <cmath>
#include <functional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
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

absl::string_view CostToString(Cost activation) {
  return kCostStr[static_cast<int32_t>(activation)];
}

absl::StatusOr<Cost> CostFromString(std::string cost_str) {
  using enum Cost;
  for (int32_t i = 0; i < kCostStr.size(); i++) {
    if (cost_str == kCostStr[i]) {
      return static_cast<Cost>(i);
    }
  }
  return absl::InvalidArgumentError(absl::StrCat(
        "Unrecognized cost: ", cost_str,
        ". Available types: ", absl::StrJoin(kCostStr, ", ")));
}
