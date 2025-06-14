#ifndef SRC_COMMON_WORKER_POOL_H_
#define SRC_COMMON_WORKER_POOL_H_

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

class ThreadPool {
 public:
  explicit ThreadPool(size_t thread_count = std::thread::hardware_concurrency()) {
    threads_.reserve(thread_count);
    for (size_t i = 0; i < thread_count; i++) {
      threads_.emplace_back(&ThreadPool::ThreadPoll, this);
    }
  }

  ~ThreadPool() {
    {
      std::scoped_lock lock(work_queue_mutex_);
      terminate_ = true;
    }
    cv_.notify_all();
    for (std::thread& thread : threads_) {
      thread.join();
    }
  }

  template <typename F, typename... Args>
  std::future<std::invoke_result_t<F, Args...>>
  Push(F&& fn, Args&&... args) {
    using RetType = std::invoke_result_t<F, Args...>;
    std::future<RetType> future;
    {
        std::scoped_lock lock(work_queue_mutex_);

        auto promise = std::make_shared<std::promise<RetType>>();
        future = promise->get_future();

        work_queue_.emplace([promise = std::move(promise),
                             fn = std::forward<F>(fn),
                             ... args = std::forward<Args>(args)]() {
          if constexpr (std::is_same<RetType, void>::value) {
              std::invoke(fn, args...);
              promise->set_value();
          } else {
              promise->set_value(std::invoke(fn, args...));
          }
        });
    }
    cv_.notify_one();
    return future;
  }

 private:
  void ThreadPoll() {
    while (true) {
      std::function<void(void)> work;
      {
        std::unique_lock<std::mutex> lock(work_queue_mutex_);
        cv_.wait(lock, [&]() { return terminate_ || !work_queue_.empty(); });
        if (terminate_ && work_queue_.empty()) { break; }
        work = work_queue_.front();
        work_queue_.pop();
      }
      work();
    }
  }

  std::vector<std::thread> threads_;
  std::queue<std::function<void(void)>> work_queue_;
  std::mutex work_queue_mutex_;
  std::condition_variable cv_;
  bool terminate_;
};

#endif
