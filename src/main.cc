#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/statusor.h"
#include "absl/log/check.h"
#include "src/common/matrix.h"
#include "src/io/csv_reader.h"
#include "src/io/model_checkpoint.h"
#include "src/neural_network/neural_network.h"
#include "src/neural_network/trainer.h"

ABSL_FLAG(
    std::string, train_data_file_path, "",
    "Path to the training data.");
ABSL_FLAG(
    std::string, in_model_checkpoint_file_path, "",
    "Path to a model checkpoint to initially load into the model.");
ABSL_FLAG(
    std::string, out_model_checkpoint_file_path, "",
    "Path to where to write the final, trained model checkpoint.");

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  CHECK(!absl::GetFlag(FLAGS_train_data_file_path).empty())
    << "Must provide --train_data_file_path.";
  CHECK(!absl::GetFlag(FLAGS_out_model_checkpoint_file_path).empty())
    << "Must provide --out_model_checkpoint_file_path.";

  absl::StatusOr<CsvReader> reader = CsvReader::Open(
      absl::GetFlag(FLAGS_train_data_file_path));
  CHECK_OK(reader);

  Parameters params = {
    .layer_sizes = {784, 200, 80, 10},
    .intermed_activation = Activation::SIGMOID,
    .output_activation = Activation::SOFTMAX,
    .cost = Cost::MEAN_SQUARED,
    .learn_rate = 0.05,
    .momentum = 0.5,
    .regularization = 0.0,
    .num_threads = std::thread::hardware_concurrency(),
    .batch_size = 10,
    .num_epochs = 1,
  };

  NeuralNetwork neural_network = [&]() {
    if (!absl::GetFlag(FLAGS_in_model_checkpoint_file_path).empty()) {
      absl::StatusOr<protos::ModelCheckpoint> starting_checkpoint =
        ReadModelCheckpoint(absl::GetFlag(FLAGS_in_model_checkpoint_file_path));
      CHECK_OK(starting_checkpoint);
      absl::StatusOr<NeuralNetwork> neural_network =
        NeuralNetwork::FromCheckpoint(*starting_checkpoint, params);
      CHECK_OK(neural_network);
      return *neural_network;
    }
    return NeuralNetwork::Random(params);
  }();
  Train(neural_network, std::move(params), std::move(*reader));
  CHECK_OK(WriteModelCheckpoint(
      absl::GetFlag(FLAGS_out_model_checkpoint_file_path),
      neural_network.ToCheckpoint()));

  return 0;
}
