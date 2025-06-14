#ifndef SRC_ACTIVATION_H_
#define SRC_ACTIVATION_H_

#include <functional>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "src/common/matrix.h"
#include "src/protos/model_checkpoint.pb.h"

std::function<Matrix(const Matrix&)> GetActivation(protos::Activation activation);
std::function<Matrix(const Matrix&)> GetActivationDeriv(protos::Activation activation);
absl::string_view ActivationToString(protos::Activation activation);
absl::StatusOr<protos::Activation> ActivationFromString(std::string activation_str);

#endif
