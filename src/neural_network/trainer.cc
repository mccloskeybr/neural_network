#include "src/neural_network/trainer.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <future>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

#include "src/params.h"
#include "src/common/matrix.h"
#include "src/io/csv_reader.h"
#include "src/neural_network/neural_network.h"

struct Stats {
  explicit Stats(const char* name) :
    name_(name),
    total_correct_inferences_(0),
    total_inferences_(0) {}

  std::string ToString() {
    std::stringstream stream;
    stream <<
      name_ << ":" << std::endl <<
      "total_correct_inferences: " << total_correct_inferences_ << std::endl <<
      "total_inferences: " << total_inferences_ << std::endl <<
      "accuracy: " << (((float) total_correct_inferences_) / total_inferences_) << std::endl <<
      std::endl;
    return stream.str();
  }

  const char* name_;
  int32_t total_correct_inferences_;
  int32_t total_inferences_;
};

struct WorkerOutput {
  WorkerOutput() : stats(Stats("WORKER")) {}
  Stats stats;
  std::vector<std::vector<std::pair<Matrix, Matrix>>> gradients;
};

void TrainPartition(
    const NeuralNetwork& neural_network,
    std::vector<std::pair<uint32_t, Matrix>> samples,
    std::promise<WorkerOutput> result) {
  WorkerOutput worker_output;
  worker_output.gradients.reserve(neural_network.LayersCount());
  for (int32_t i = 0; i < samples.size(); i++) {
    uint32_t expected_class = samples[i].first;
    Matrix input = std::move(samples[i].second);

    // TODO: apply this directly to file data, this is specific to the MNIST data set.
    input = input.Map([](float x) { return x / 255.0f; } );
    Matrix expected_output = Matrix(1, 10);
    expected_output.MutableElementAt(0, expected_class) = 1.0f;

    NeuralNetwork::NetworkLearnCache cache = {};
    Matrix model_output = neural_network.FeedForward(std::move(input), &cache);
    worker_output.stats.total_correct_inferences_ =
      (model_output.Classify() == expected_output.Classify());
    worker_output.stats.total_inferences_++;
    worker_output.gradients.emplace_back(neural_network.BackPropagate(
        std::move(model_output), std::move(expected_output), &cache));
  }
  result.set_value(worker_output);
}

void TrainEpoch(
    Parameters params, NeuralNetwork& neural_network, CsvReader& reader) {
  auto epoch_stats = Stats("EPOCH");

  int32_t ideal_partition_size = std::ceil(((float) params.batch_size) / params.num_threads);
  ASSERT(ideal_partition_size > 0);
  std::vector<std::pair<uint32_t, Matrix>> partition;
  partition.reserve(ideal_partition_size);

  std::vector<std::pair<uint32_t, Matrix>> batch = reader.GetNextBatchSample(params.batch_size);
  while (batch.size() > 0) { // NOTE: while there is still file data
    auto batch_stats = Stats("BATCH");

    std::vector<std::thread> worker_threads;
    worker_threads.reserve(params.batch_size);
    std::vector<std::future<WorkerOutput>> worker_outputs;
    worker_outputs.reserve(params.batch_size);

    while (batch.size() > 0) { // NOTE: while processing a given batch
      partition.clear();
      int32_t partition_size = std::min(ideal_partition_size, (int32_t) batch.size());
      for (int32_t i = 0; i < partition_size; i++) {
        partition.emplace_back(batch.back());
        batch.pop_back();
      }
      std::promise<WorkerOutput> worker_promise;
      worker_outputs.push_back(worker_promise.get_future());
      worker_threads.push_back(std::thread(
            TrainPartition, neural_network, partition, std::move(worker_promise)));
    }

    std::vector<std::pair<Matrix, Matrix>> gradients_accum;
    gradients_accum.reserve(neural_network.LayersCount());
    for (int32_t i = 0; i < neural_network.LayersCount(); i++) {
      const Layer& layer = neural_network.GetLayer(i);
      gradients_accum.emplace_back(std::make_pair(
            Matrix(layer.InputSize(), layer.OutputSize()),
            Matrix(1, layer.OutputSize())));
    }

    for (std::thread& worker_thread : worker_threads) { worker_thread.join(); }
    for (int32_t i = 0; i < worker_outputs.size(); i++) {
      WorkerOutput worker_output = worker_outputs[i].get();
      batch_stats.total_correct_inferences_ += worker_output.stats.total_correct_inferences_;
      batch_stats.total_inferences_ += worker_output.stats.total_inferences_;
      for (std::vector<std::pair<Matrix, Matrix>>& gradients : worker_output.gradients) {
        for (int32_t j = 0; j < gradients.size(); j++) {
          // TODO: operator+= for matrix
          gradients_accum[j].first = gradients_accum[j].first + gradients[j].first;
          gradients_accum[j].second = gradients_accum[j].second + gradients[j].second;
        }
      }
    }
    neural_network.ApplyGradients(gradients_accum);

    epoch_stats.total_correct_inferences_ += batch_stats.total_correct_inferences_;
    epoch_stats.total_inferences_ += batch_stats.total_inferences_;
    std::cout << epoch_stats.ToString();

    batch = reader.GetNextBatchSample(params.batch_size);
  }

  std::cout << epoch_stats.ToString();
}

NeuralNetwork Train(Parameters params, CsvReader reader) {
  auto neural_network = NeuralNetwork::Random(params);
  for (int32_t i = 0; i < params.num_epochs; i++) {
    TrainEpoch(params, neural_network, reader);
    reader.Reset();
  }
  return neural_network;
}
