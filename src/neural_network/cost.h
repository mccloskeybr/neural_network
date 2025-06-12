#ifndef SRC_COST_H_
#define SRC_COST_H_

#include <functional>

enum class Cost {
  MEAN_SQUARED,
};

std::function<double(double, double)> GetCost(Cost cost);
std::function<double(double, double)> GetCostDeriv(Cost cost);

#endif
