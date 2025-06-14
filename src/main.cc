#include <string>
#include <thread>
#include <vector>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "src/common/matrix.h"
#include "src/io/csv_reader.h"
#include "src/io/model_checkpoint.h"
#include "src/neural_network/params.h"
#include "src/neural_network/neural_network.h"
#include "src/neural_network/trainer.h"

// Input and output data file paths
ABSL_FLAG(
    std::string, train_data_file_path, "",
    "Path to the training dataset.");
ABSL_FLAG(
    std::string, test_data_file_path, "",
    "Path to the test dataset.");
ABSL_FLAG(
    std::string, out_model_checkpoint_file_path, "",
    "Path to where to write the final, trained model checkpoint.");

// Existing model checkpoint file path
ABSL_FLAG(
    std::string, in_model_checkpoint_file_path, "",
    "Path to a model checkpoint to initially load into the model.");

// New model generation flags
ABSL_FLAG(
    std::vector<std::string>, layer_sizes, {},
    "Integer vector of layer sizes that define an entirely new model's shape.");
ABSL_FLAG(
    std::string, intermediate_activation,
    std::string(ActivationToString(protos::Activation::SIGMOID)),
    "Intermediate / hidden layer activation function.");
ABSL_FLAG(
    std::string, output_activation,
    std::string(ActivationToString(protos::Activation::SOFTMAX)),
    "Output layer activation function.");

// Training parameters
ABSL_FLAG(
    std::string, cost,
    std::string(CostToString(Cost::MEAN_SQUARED)),
    "Cost function.");
ABSL_FLAG(
    double, learn_rate, 0.05,
    "Learn rate, the initial step size for gradient descent.");
ABSL_FLAG(
    double, momentum, 0.5,
    "Momentum, influences the gradient descent step size / resistance.");
ABSL_FLAG(
    double, regularization, 0.0,
    "Regularization, helps combat overfitting");
ABSL_FLAG(
    uint32_t, num_threads, std::thread::hardware_concurrency(),
    "The number of threads to include in the training thread pool.");
ABSL_FLAG(
    uint32_t, train_batch_size, 12,
    "The number of samples to learn on concurrently.");
ABSL_FLAG(
    uint32_t, test_batch_size, 128,
    "The number of samples to test concurrently.");
ABSL_FLAG(
    uint32_t, num_epochs, 1,
    "The number of times the training data will be iterated through");

int main(int argc, char* argv[]) {
  absl::InitializeLog();
  absl::ParseCommandLine(argc, argv);

  CHECK(!absl::GetFlag(FLAGS_train_data_file_path).empty())
    << "Must provide --train_data_file_path.";
  CHECK(!absl::GetFlag(FLAGS_test_data_file_path).empty())
    << "Must provide --test_data_file_path.";
  CHECK(!absl::GetFlag(FLAGS_out_model_checkpoint_file_path).empty())
    << "Must provide --out_model_checkpoint_file_path.";

  auto neural_network = [&]() -> absl::StatusOr<NeuralNetwork> {
    if (!absl::GetFlag(FLAGS_in_model_checkpoint_file_path).empty()) {
      LOG(INFO) << "Using model checkpoint path: "
        << absl::GetFlag(FLAGS_in_model_checkpoint_file_path);

      absl::StatusOr<protos::ModelCheckpoint> starting_checkpoint =
        ReadModelCheckpoint(absl::GetFlag(FLAGS_in_model_checkpoint_file_path));
      CHECK_OK(starting_checkpoint);
      absl::StatusOr<NeuralNetwork> neural_network =
        NeuralNetwork::FromCheckpoint(*starting_checkpoint);
      return neural_network;

    } else if (!absl::GetFlag(FLAGS_layer_sizes).empty()) {
      LOG(INFO) << "Generating new model with: { "
        << "dimensions: [ " << absl::StrJoin(absl::GetFlag(FLAGS_layer_sizes), ", ") << " ]"
        << ", intermed_activation: " << absl::GetFlag(FLAGS_intermediate_activation)
        << ", output_activation: " << absl::GetFlag(FLAGS_output_activation)
        << " }.";
      std::vector<int32_t> layer_sizes;
      layer_sizes.reserve(absl::GetFlag(FLAGS_layer_sizes).size());
      for (const std::string& layer_size_str : absl::GetFlag(FLAGS_layer_sizes)) {
        int32_t layer_size;
        if (!absl::SimpleAtoi(layer_size_str, &layer_size)) {
          return absl::InvalidArgumentError( absl::StrCat("Unable to parse layer size: ", layer_size_str));
        }
        layer_sizes.push_back(layer_size);
      }
      absl::StatusOr<protos::Activation> intermed_activation =
        ActivationFromString(absl::GetFlag(FLAGS_intermediate_activation));
      absl::StatusOr<protos::Activation> output_activation =
        ActivationFromString(absl::GetFlag(FLAGS_output_activation));
      CHECK_OK(intermed_activation);
      CHECK_OK(output_activation);

      return NeuralNetwork::Random(std::move(layer_sizes), *intermed_activation, *output_activation);

    }

    return absl::InvalidArgumentError(
        "Must provide one of "
        "{ --in_model_checkpoint_file_path } or "
        "{ --layer_sizes, --intermediate_activation, --output_activation }.");
  }();
  CHECK_OK(neural_network);

  absl::StatusOr<Cost> cost =
    CostFromString(absl::GetFlag(FLAGS_cost));
  CHECK_OK(cost);
  TrainParameters train_params = {
    .cost = *cost,
    .learn_rate = absl::GetFlag(FLAGS_learn_rate),
    .momentum = absl::GetFlag(FLAGS_momentum),
    .regularization = absl::GetFlag(FLAGS_regularization),
    .num_threads = absl::GetFlag(FLAGS_num_threads),
    .num_epochs = absl::GetFlag(FLAGS_num_epochs),
    .train_batch_size = absl::GetFlag(FLAGS_train_batch_size),
    .test_batch_size = absl::GetFlag(FLAGS_test_batch_size),
  };
  CHECK_OK(Train(
      *neural_network, train_params,
      absl::GetFlag(FLAGS_train_data_file_path),
      absl::GetFlag(FLAGS_test_data_file_path),
      absl::GetFlag(FLAGS_out_model_checkpoint_file_path)));

  return 0;
}
