#ifndef SRC_COST_H_
#define SRC_COST_H_

#include <functional>

#include "src/common/matrix.h"

enum class Cost {
  MEAN_SQUARED,
};

std::function<Matrix(const Matrix&, const Matrix&)> GetCost(Cost cost);
std::function<Matrix(const Matrix&, const Matrix&)> GetCostDeriv(Cost cost);

#endif
