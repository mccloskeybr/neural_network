#include "activation.h"

#include <cmath>
#include <functional>

#include "assert.h"

float Sigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

float SigmoidDeriv(float x) {
  float y = Sigmoid(x);
  return y * (1.0f - y);
}

std::function<float(float)> GetActivation(Activation activation) {
  using enum Activation;
  switch (activation) {
    case SIGMOID: { return Sigmoid; }
    default: { UNREACHABLE(); return Sigmoid; }
  }
}

std::function<float(float)> GetActivationDeriv(Activation activation) {
  using enum Activation;
  switch (activation) {
    case SIGMOID: { return SigmoidDeriv; }
    default: { UNREACHABLE(); return SigmoidDeriv; }
  }
}
