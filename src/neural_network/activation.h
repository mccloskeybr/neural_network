#ifndef SRC_ACTIVATION_H_
#define SRC_ACTIVATION_H_

#include <functional>

enum class Activation {
  SIGMOID,
};

std::function<float(float)> GetActivation(Activation activation);
std::function<float(float)> GetActivationDeriv(Activation activation);

#endif
