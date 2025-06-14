#include "src/neural_network/activation.h"

#include <cmath>
#include <functional>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "src/common/matrix.h"
#include "src/protos/model_checkpoint.pb.h"

Matrix Sigmoid(const Matrix& m) {
  DCHECK(m.RowCount() == 1);
  Matrix result = m;
  for (int32_t c = 0; c < result.ColCount(); c++) {
    double& e = result.MutableElementAt(0, c);
    e = (1.0f / (1.0f + std::exp(-e)));
  }
  return result;
}

Matrix SigmoidDeriv(const Matrix& m) {
  DCHECK(m.RowCount() == 1);
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
  DCHECK(m.RowCount() == 1);
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
  DCHECK(m.RowCount() == 1);
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
  DCHECK(m.RowCount() == 1);
  Matrix result = m;
  for (int32_t c = 0; c < result.ColCount(); c++) {
    double& e = result.MutableElementAt(0, c);
    double e_2 = std::exp(2 * e);
    e = ((e_2 - 1) / (e_2 + 1));
  }
  return result;
}

Matrix TanHDeriv(const Matrix& m) {
  DCHECK(m.RowCount() == 1);
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
  DCHECK(m.RowCount() == 1);
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
  DCHECK(m.RowCount() == 1);
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

std::function<Matrix(const Matrix&)> GetActivation(protos::Activation activation) {
  switch (activation) {
    case protos::Activation::SIGMOID: { return Sigmoid; }
    case protos::Activation::RELU: { return ReLU; }
    case protos::Activation::TANH: { return TanH; }
    case protos::Activation::SOFTMAX: { return Softmax; }
    default: { CHECK(false); return Sigmoid; }
  }
}

std::function<Matrix(const Matrix&)> GetActivationDeriv(protos::Activation activation) {
  switch (activation) {
    case protos::Activation::SIGMOID: { return SigmoidDeriv; }
    case protos::Activation::RELU: { return ReLUDeriv; }
    case protos::Activation::TANH: { return TanHDeriv; }
    case protos::Activation::SOFTMAX: { return SoftmaxDeriv; }
    default: { CHECK(false); return SigmoidDeriv; }
  }
}

absl::string_view ActivationToString(protos::Activation activation) {
  absl::string_view activation_str = protos::Activation_Name(activation);
  DCHECK(!activation_str.empty());
  return activation_str;
}

absl::StatusOr<protos::Activation> ActivationFromString(std::string activation_str) {
  protos::Activation activation;
  if (protos::Activation_Parse(activation_str, &activation)) { return activation; }
  return absl::InvalidArgumentError(absl::StrCat(
        "Failed to parse activation type from: ", activation_str, "."));
}
