// TLDR; this is experimenting with different matrix multiplication implementations,
// since that is by far the hottest path in neural network usage.
//
// This tests the naive approach, a more cache friendly implementation w/ loop reordering,
// and a tiled matrix multiplication.
//
// Results show that the cache friendly implementation / loop reordering strategy worked the
// best on my machine. Likely that the matrices are too small? Tiled matmul also benefits from
// multithreading, the current strategy this lib takes is going horizontal on batches rather than
// individual matmuls, so that could potentially be reworked as well. Could just be that
// tiled matmul is better suited for accelerated hardware rather than CPUs anyway.

#include <cstdint>
#include <vector>
#include <utility>

#include "benchmark/benchmark.h"

constexpr int32_t kNumRepetitions = 3;
constexpr int32_t kNumIterations = 100;

struct Matrix {
  explicit Matrix(int32_t row_count, int32_t col_count) :
    row_count_(row_count), col_count_(col_count),
    elements_(std::vector<double>(row_count_ * col_count_)) {}

  double& MutableElementAt(int32_t r, int32_t c) { return elements_[r * col_count_ + c]; }
  const double& ElementAt(int32_t r, int32_t c) const { return elements_[r * col_count_ + c]; }

  Matrix NaiveMultiply(const Matrix& b) const;
  Matrix CacheMultiply(const Matrix& b) const;
  Matrix TiledMultiply(const Matrix& b) const;

  int32_t row_count_;
  int32_t col_count_;
  std::vector<double> elements_;
};

Matrix Matrix::NaiveMultiply(const Matrix& other) const {
  Matrix result(row_count_, other.col_count_);
  for (int32_t i = 0; i < row_count_; i++) {
    for (int32_t j = 0; j < other.col_count_; j++) {
      double sum = result.ElementAt(i, j);
      for (int32_t k = 0; k < col_count_; k++) {
        sum += ElementAt(i, k) * other.ElementAt(k, k);
      }
      result.MutableElementAt(i, j) = sum;
    }
  }
  return result;
}

Matrix Matrix::CacheMultiply(const Matrix& other) const {
  Matrix result(row_count_, other.col_count_);
  for (int32_t i = 0; i < row_count_; i++) {
    for (int32_t k = 0; k < col_count_; k++) {
      for (int32_t j = 0; j < other.col_count_; j++) {
        result.MutableElementAt(i, j) += ElementAt(i, k) * other.ElementAt(k, j);
      }
    }
  }
  return result;
}

// https://stackoverflow.com/questions/59009628/tiled-matrix-multiplication-using-avx
// https://stackoverflow.com/questions/15829223/loop-tiling-blocking-for-large-dense-matrix-multiplication
// Other improvements are possible, could manually hoist variables & play with loop reordering...
constexpr int32_t kCacheLineSizeBytes = 64;
constexpr int32_t kBlockSize = kCacheLineSizeBytes / sizeof(double);
Matrix Matrix::TiledMultiply(const Matrix& other) const {
  Matrix result(row_count_, other.col_count_);
  for (int32_t i_0 = 0; i_0 < row_count_; i_0 += kBlockSize) {
    int32_t i_max = std::min(i_0 + kBlockSize, row_count_);
    for (int32_t k_0 = 0; k_0 < col_count_; k_0 += kBlockSize) {
      int32_t k_max = std::min(k_0 + kBlockSize, col_count_);
      for (int32_t j_0 = 0; j_0 < other.col_count_; j_0 += kBlockSize) {
        int32_t j_max = std::min(j_0 + kBlockSize, other.col_count_);
        for (int32_t i = i_0; i < i_max; i++) {
          for (int32_t j = j_0; j < j_max; j++) {
            for (int32_t k = k_0; k < k_max; k++) {
              result.MutableElementAt(i, j) += ElementAt(i, k) * other.ElementAt(k, j);
            }
          }
        }
      }
    }
  }
  return result;
}

std::pair<Matrix, Matrix> BuildTestMatrices(int32_t x, int32_t y) {
  std::vector<double> a_elem = std::vector<double>();
  a_elem.reserve(x * y);
  std::vector<double> b_elem = std::vector<double>();
  b_elem.reserve(y * x);
  for (int32_t i = 0; i < x * y; i++) {
    a_elem.push_back(i);
    b_elem.push_back(i * 10);
  }
  Matrix a(x, y);
  a.elements_ = std::move(a_elem);
  Matrix b(y, x);
  b.elements_ = std::move(b_elem);
  return std::make_pair(std::move(a), std::move(b));
}

void BM_NaiveMultiply(benchmark::State& state) {
  const auto [a, b] = BuildTestMatrices(state.range(0), state.range(1));
  for (auto _ : state) {
    for (int64_t i = 0; i < kNumIterations; i++) {
      Matrix c = a.NaiveMultiply(b);
    }
  }
}
BENCHMARK(BM_NaiveMultiply)
  ->Repetitions(kNumRepetitions)
  ->DisplayAggregatesOnly(true)
  ->Args({32, 32})
  ->Args({256, 256})
  ->Args({512, 512});
  ->Args({256, 512})
  ->Args({512, 256});

void BM_CacheMultiply(benchmark::State& state) {
  auto [a, b] = BuildTestMatrices(state.range(0), state.range(1));
  for (auto _ : state) {
    for (int64_t i = 0; i < kNumIterations; i++) {
      Matrix c = a.CacheMultiply(b);
    }
  }
}
BENCHMARK(BM_CacheMultiply)
  ->Repetitions(kNumRepetitions)
  ->DisplayAggregatesOnly(true)
  ->Args({32, 32})
  ->Args({256, 256})
  ->Args({512, 512});
  ->Args({256, 512})
  ->Args({512, 256});

void BM_TiledMultiply(benchmark::State& state) {
  auto [a, b] = BuildTestMatrices(state.range(0), state.range(1));
  for (auto _ : state) {
    for (int64_t i = 0; i < kNumIterations; i++) {
      Matrix c = a.TiledMultiply(b);
    }
  }
}
BENCHMARK(BM_TiledMultiply)
  ->Repetitions(kNumRepetitions)
  ->DisplayAggregatesOnly(true)
  ->Args({32, 32})
  ->Args({256, 256})
  ->Args({512, 512});
  ->Args({256, 512})
  ->Args({512, 256});

BENCHMARK_MAIN();
