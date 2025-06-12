#ifndef SRC_ACTIVATION_H_
#define SRC_ACTIVATION_H_

#include <functional>

enum class Activation {
  SIGMOID,
};

std::function<double(double)> GetActivation(Activation activation);
std::function<double(double)> GetActivationDeriv(Activation activation);

#endif
