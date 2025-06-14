#include "src/neural_network/trainer.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <future>
#include <string>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/common/matrix.h"
#include "src/common/thread_pool.h"
#include "src/io/csv_reader.h"
#include "src/io/model_checkpoint.h"
#include "src/neural_network/neural_network.h"

struct Stats {
  Stats() : total_correct_inferences_(0), total_inferences_(0), num_batches_(0) {}
  std::string ToString() {
    return absl::StrCat(
        "{ num_batches: ", num_batches_,
        ", total_correct_inferences: ", total_correct_inferences_,
        ", total_inferences: ", total_inferences_,
        ", accuracy: ", (((double) total_correct_inferences_) / total_inferences_),
        " }");
  }
  int32_t total_correct_inferences_;
  int32_t total_inferences_;
  int32_t num_batches_;
};

struct WorkerOutput {
  Stats stats;
  std::vector<std::vector<std::pair<Matrix, Matrix>>> gradients;
};

WorkerOutput TrainPartition(
    const TrainParameters& params,
    const NeuralNetwork& neural_network,
    std::vector<std::pair<uint32_t, Matrix>> samples) {
  WorkerOutput worker_output;
  worker_output.gradients.reserve(neural_network.LayersCount());
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
    Matrix model_output = neural_network.FeedForward(input, &cache);
    worker_output.stats.total_correct_inferences_ +=
      (model_output.Classify() == expected_output.Classify());
    worker_output.stats.total_inferences_++;
    worker_output.gradients.emplace_back(neural_network.BackPropagate(
          params, &cache, model_output, expected_output));
  }
  return worker_output;
}

Stats TrainEpoch(
    const TrainParameters& params, NeuralNetwork& neural_network,
    CsvReader& train_data, ThreadPool& thread_pool) {
  Stats stats;

  int32_t ideal_partition_size = std::ceil(((double) params.train_batch_size) / params.num_threads);
  DCHECK(ideal_partition_size > 0);

  std::vector<std::pair<uint32_t, Matrix>> batch = train_data.GetNextBatchSample(params.train_batch_size);
  while (batch.size() > 0) { // NOTE: while there is still file data

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
          TrainPartition, params, neural_network, std::move(sample_partition));
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
      stats.total_correct_inferences_ += worker_output.stats.total_correct_inferences_;
      stats.total_inferences_ += worker_output.stats.total_inferences_;
      for (std::vector<std::pair<Matrix, Matrix>>& gradients : worker_output.gradients) {
        for (int32_t j = 0; j < gradients.size(); j++) {
          gradients_accum[j].first += gradients[j].first;
          gradients_accum[j].second += gradients[j].second;
        }
      }
    }
    neural_network.ApplyGradients(params, std::move(gradients_accum));

    stats.num_batches_++;
    batch = train_data.GetNextBatchSample(params.train_batch_size);
    LOG_EVERY_N_SEC(INFO, 15) << "Epoch progress: " << stats.ToString();
  }

  return stats;
}

Stats TestPartition(
    const NeuralNetwork& neural_network,
    std::vector<std::pair<uint32_t, Matrix>> samples) {
  Stats stats;
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
    Matrix model_output = neural_network.Infer(input);

    stats.total_correct_inferences_ +=
      (model_output.Classify() == expected_output.Classify());
    stats.total_inferences_++;
  }
  return stats;
}

Stats Test(
    const TrainParameters& params, const NeuralNetwork& neural_network,
    CsvReader& test_data, ThreadPool& thread_pool) {
  Stats stats;

  int32_t ideal_partition_size = std::ceil(((double) params.test_batch_size) / params.num_threads);
  DCHECK(ideal_partition_size > 0);

  std::vector<std::pair<uint32_t, Matrix>> batch = test_data.GetNextBatchSample(params.test_batch_size);
  while (batch.size() > 0) { // NOTE: while there is still file data
    // NOTE: enqueue batch work
    std::vector<std::future<Stats>> all_worker_stats;
    all_worker_stats.reserve(params.num_threads);
    while (batch.size() > 0) {
      std::vector<std::pair<uint32_t, Matrix>> sample_partition;
      int32_t sample_partition_size = std::min(ideal_partition_size, (int32_t) batch.size());
      sample_partition.reserve(std::min(sample_partition_size, (int32_t) batch.size()));
      for (int32_t i = 0; i < sample_partition_size; i++) {
        sample_partition.push_back(std::move(batch.back()));
        batch.pop_back();
      }

      std::future<Stats> future = thread_pool.Push(
          TestPartition, neural_network, std::move(sample_partition));
      all_worker_stats.push_back(std::move(future));
    }

    // NOTE: read work output
    for (std::future<Stats>& worker_stats_future : all_worker_stats) {
      worker_stats_future.wait();
      Stats worker_stats = worker_stats_future.get();
      stats.total_correct_inferences_ += worker_stats.total_correct_inferences_;
      stats.total_inferences_ += worker_stats.total_inferences_;
    }

    stats.num_batches_++;
    batch = test_data.GetNextBatchSample(params.test_batch_size);
  }

  return stats;
}

absl::Status Train(
    NeuralNetwork& neural_network, const TrainParameters& params,
    std::string train_data_file_path, std::string test_data_file_path,
    std::string out_model_checkpoint_file_path) {
  absl::StatusOr<CsvReader> train_data = CsvReader::Open(train_data_file_path);
  if (!train_data.ok()) { return train_data.status(); }
  absl::StatusOr<CsvReader> test_data = CsvReader::Open(test_data_file_path);
  if (!test_data.ok()) { return test_data.status(); }

  LOG(INFO) << "Using training params: " << params.ToString();
  auto thread_pool = ThreadPool(params.num_threads);
  for (int32_t i = 0; i < params.num_epochs; i++) {
    LOG(INFO) << "Epoch " << (i + 1) << " of " << params.num_epochs << ": Starting training...";
    train_data->Reset();
    Stats train_stats = TrainEpoch(params, neural_network, *train_data, thread_pool);
    LOG(INFO) << "Epoch " << (i + 1) << " of " << params.num_epochs << ": Train score: " << train_stats.ToString();

    test_data->Reset();
    Stats test_stats = Test(params, neural_network, *test_data, thread_pool);
    LOG(INFO) << "Epoch " << (i + 1) << " of " << params.num_epochs << ": Test score : " << test_stats.ToString();

    LOG(INFO) << "Saving model checkpoint to: " << out_model_checkpoint_file_path << ".";
    absl::Status checkpoint_status = WriteModelCheckpoint(
        out_model_checkpoint_file_path, neural_network.ToCheckpoint());
    if (!checkpoint_status.ok()) { return checkpoint_status; }
  }

  return absl::OkStatus();
}
