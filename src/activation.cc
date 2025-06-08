#include "activation.h"

#include <cassert>
#include <cmath>
#include <functional>

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
    default: { assert(false); return Sigmoid; }
  }
}

std::function<float(float)> GetActivationDeriv(Activation activation) {
  using enum Activation;
  switch (activation) {
    case SIGMOID: { return SigmoidDeriv; }
    default: { assert(false); return SigmoidDeriv; }
  }
}
