#ifndef SRC_COST_H_
#define SRC_COST_H_

#include <functional>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "src/common/matrix.h"

enum class Cost {
  MEAN_SQUARED,
};

static const std::vector<absl::string_view> kCostStr {
  "MEAN_SQUARED",
};

std::function<Matrix(const Matrix&, const Matrix&)> GetCost(Cost cost);
std::function<Matrix(const Matrix&, const Matrix&)> GetCostDeriv(Cost cost);
absl::string_view CostToString(Cost activation);
absl::StatusOr<Cost> CostFromString(std::string activation_str);

#endif
