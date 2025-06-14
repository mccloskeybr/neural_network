load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
  name = "assert",
  hdrs = ["assert.h"],
  deps = [],
)

cc_library(
  name = "matrix",
  hdrs = ["matrix.h"],
  srcs = ["matrix.cc"],
  deps = [
    "@abseil-cpp//absl/log:check",
  ],
)

cc_test(
  name = "matrix_test",
  srcs = ["matrix_test.cc"],
  deps = [
    ":matrix",
    "@googletest//:gtest",
    "@googletest//:gtest_main",
  ],
)

cc_library(
  name = "thread_pool",
  hdrs = ["thread_pool.h"],
  deps = [],
)
