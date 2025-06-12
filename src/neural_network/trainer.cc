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
#include "src/common/thread_pool.h"
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
      "accuracy: " << (((double) total_correct_inferences_) / total_inferences_) << std::endl <<
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

WorkerOutput TrainPartition(
    const NeuralNetwork* neural_network,
    std::vector<std::pair<uint32_t, Matrix>> samples) {
  WorkerOutput worker_output;
  worker_output.gradients.reserve(neural_network->LayersCount());
  for (int32_t i = 0; i < samples.size(); i++) {
    uint32_t expected_class = samples[i].first;
    Matrix& input = samples[i].second;

    // TODO/SPEEDUP: apply this directly to file data, this is specific to the MNIST data set.
    for (int32_t r = 0; r < input.RowCount(); r++) {
      for (int32_t c = 0; c < input.ColCount(); c++) {
        double& x = input.MutableElementAt(r, c);
        x /= 255.0;
      }
    }
    Matrix expected_output = Matrix(1, 10);
    expected_output.MutableElementAt(0, expected_class) = 1.0f;

    NeuralNetwork::NetworkLearnCache cache = {};
    Matrix model_output = neural_network->FeedForward(input, &cache);
    worker_output.stats.total_correct_inferences_ =
      (model_output.Classify() == expected_output.Classify());
    worker_output.stats.total_inferences_++;
    worker_output.gradients.emplace_back(neural_network->BackPropagate(
        model_output, expected_output, &cache));
  }
  return worker_output;
}

void TrainEpoch(
    Parameters params, NeuralNetwork& neural_network, CsvReader& reader,
    ThreadPool& thread_pool) {
  auto epoch_stats = Stats("EPOCH");

  int32_t ideal_partition_size = std::ceil(((double) params.batch_size) / params.num_threads);
  ASSERT(ideal_partition_size > 0);

  std::vector<std::pair<uint32_t, Matrix>> batch = reader.GetNextBatchSample(params.batch_size);
  while (batch.size() > 0) { // NOTE: while there is still file data
    auto batch_stats = Stats("BATCH");

    // NOTE: enqueue batch work
    std::vector<std::future<WorkerOutput>> worker_output_futures;
    worker_output_futures.reserve(params.num_threads);
    while (batch.size() > 0) {
      std::vector<std::pair<uint32_t, Matrix>> sample_partition;
      int32_t sample_partition_size = std::min(ideal_partition_size, (int32_t) batch.size());
      sample_partition.reserve(std::min(sample_partition_size, (int32_t) batch.size()));
      for (int32_t i = 0; i < sample_partition_size; i++) {
        sample_partition.push_back(std::move(batch.back()));
        batch.pop_back();
      }

      std::future<WorkerOutput> future = thread_pool.Push(
          TrainPartition, &neural_network, std::move(sample_partition));
      worker_output_futures.push_back(std::move(future));
    }

    std::vector<std::pair<Matrix, Matrix>> gradients_accum;
    gradients_accum.reserve(neural_network.LayersCount());
    for (int32_t i = 0; i < neural_network.LayersCount(); i++) {
      const Layer& layer = neural_network.GetLayer(i);
      gradients_accum.emplace_back(std::make_pair(
            Matrix(layer.InputSize(), layer.OutputSize()),
            Matrix(1, layer.OutputSize())));
    }
    for (std::future<WorkerOutput>& worker_output_future : worker_output_futures) {
      worker_output_future.wait();
      WorkerOutput worker_output = worker_output_future.get();
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
  auto thread_pool = ThreadPool(params.num_threads);
  auto neural_network = NeuralNetwork::Random(params);
  for (int32_t i = 0; i < params.num_epochs; i++) {
    TrainEpoch(params, neural_network, reader, thread_pool);
    reader.Reset();
  }
  return neural_network;
}
