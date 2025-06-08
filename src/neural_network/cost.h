#ifndef SRC_COST_H_
#define SRC_COST_H_

#include <functional>

enum class Cost {
  MEAN_SQUARED,
};

std::function<float(float, float)> GetCost(Cost cost);
std::function<float(float, float)> GetCostDeriv(Cost cost);

#endif
