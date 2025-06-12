#ifndef SRC_ACTIVATION_H_
#define SRC_ACTIVATION_H_

#include <functional>

#include "src/common/matrix.h"

enum class Activation {
  SIGMOID,
  RELU,
  TANH,
  SOFTMAX,
};

std::function<Matrix(const Matrix&)> GetActivation(Activation activation);
std::function<Matrix(const Matrix&)> GetActivationDeriv(Activation activation);

#endif
