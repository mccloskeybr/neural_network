#include "src/neural_network/activation.h"

#include <cmath>
#include <functional>

#include "src/common/assert.h"

double Sigmoid(double x) {
  return 1.0f / (1.0f + std::exp(-x));
}

double SigmoidDeriv(double x) {
  double y = Sigmoid(x);
  return y * (1.0f - y);
}

std::function<double(double)> GetActivation(Activation activation) {
  using enum Activation;
  switch (activation) {
    case SIGMOID: { return Sigmoid; }
    default: { UNREACHABLE(); return Sigmoid; }
  }
}

std::function<double(double)> GetActivationDeriv(Activation activation) {
  using enum Activation;
  switch (activation) {
    case SIGMOID: { return SigmoidDeriv; }
    default: { UNREACHABLE(); return SigmoidDeriv; }
  }
}
