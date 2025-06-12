#include "src/neural_network/activation.h"

#include <cmath>
#include <functional>

#include "src/common/assert.h"
#include "src/common/matrix.h"

Matrix Sigmoid(const Matrix& m) {
  ASSERT(m.RowCount() == 1);
  Matrix result = m;
  for (int32_t c = 0; c < result.ColCount(); c++) {
    double& e = result.MutableElementAt(0, c);
    e = (1.0f / (1.0f + std::exp(-e)));
  }
  return result;
}

Matrix SigmoidDeriv(const Matrix& m) {
  ASSERT(m.RowCount() == 1);
  Matrix result = m;
  for (int32_t c = 0; c < result.ColCount(); c++) {
    double& e = result.MutableElementAt(0, c);
    double a = (1.0f / (1.0f + std::exp(-e)));
    e = a * (1.0f - a);
  }
  return result;
}

// TODO: usually don't see upper bounds clamping, could investigate
Matrix ReLU(const Matrix& m) {
  ASSERT(m.RowCount() == 1);
  Matrix result = m;
  for (int32_t c = 0; c < result.ColCount(); c++) {
    double& e = result.MutableElementAt(0, c);
    if (e <= 0) { e = 0.0; }
    if (e >= 1) { e = 1.0; }
    e = e;
  }
  return result;
}

Matrix ReLUDeriv(const Matrix& m) {
  ASSERT(m.RowCount() == 1);
  Matrix result = m;
  for (int32_t c = 0; c < result.ColCount(); c++) {
    double& e = result.MutableElementAt(0, c);
    if (e <= 0) { e = 0.0; }
    if (e >= 1) { e = 0.0; }
    e = 1.0;
  }
  return result;
}

Matrix TanH(const Matrix& m) {
  ASSERT(m.RowCount() == 1);
  Matrix result = m;
  for (int32_t c = 0; c < result.ColCount(); c++) {
    double& e = result.MutableElementAt(0, c);
    double e_2 = std::exp(2 * e);
    e = ((e_2 - 1) / (e_2 + 1));
  }
  return result;
}

Matrix TanHDeriv(const Matrix& m) {
  ASSERT(m.RowCount() == 1);
  Matrix result = m;
  for (int32_t c = 0; c < result.ColCount(); c++) {
    double& e = result.MutableElementAt(0, c);
    double e_2 = std::exp(2 * e);
    double t = ((e_2 - 1) / (e_2 + 1));
    e = 1 - t * t;
  }
  return result;
}

Matrix Softmax(const Matrix& m) {
  ASSERT(m.RowCount() == 1);
  Matrix result = m;
  double exp_sum = 0.0;
  for (int32_t c = 0; c < result.ColCount(); c++) {
    const double e = result.ElementAt(0, c);
    exp_sum += std::exp(e);
  }
  for (int32_t c = 0; c < result.ColCount(); c++) {
    double& e = result.MutableElementAt(0, c);
    e = std::exp(e) / exp_sum;
  }
  return result;
}

Matrix SoftmaxDeriv(const Matrix& m) {
  ASSERT(m.RowCount() == 1);
  Matrix result = m;
  double exp_sum = 0.0;
  for (int32_t c = 0; c < result.ColCount(); c++) {
    double e = result.ElementAt(0, c);
    exp_sum += std::exp(e);
  }
  for (int32_t c = 0; c < result.ColCount(); c++) {
    double& e = result.MutableElementAt(0, c);
    double ex = std::exp(e);
    e = (((ex * exp_sum) - (ex * ex)) / (exp_sum * exp_sum));
  }
  return result;
}

std::function<Matrix(const Matrix&)> GetActivation(Activation activation) {
  using enum Activation;
  switch (activation) {
    case SIGMOID: { return Sigmoid; }
    case RELU: { return ReLU; }
    case TANH: { return TanH; }
    case SOFTMAX: { return Softmax; }
    default: { UNREACHABLE(); return Sigmoid; }
  }
}

std::function<Matrix(const Matrix&)> GetActivationDeriv(Activation activation) {
  using enum Activation;
  switch (activation) {
    case SIGMOID: { return SigmoidDeriv; }
    case RELU: { return ReLUDeriv; }
    case TANH: { return TanHDeriv; }
    case SOFTMAX: { return SoftmaxDeriv; }
    default: { UNREACHABLE(); return SigmoidDeriv; }
  }
}
