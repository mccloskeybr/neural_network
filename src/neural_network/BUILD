load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
  name = "cost",
  hdrs = ["cost.h"],
  srcs = ["cost.cc"],
  deps = [
    "//src/common:assert",
  ],
)

cc_library(
  name = "activation",
  hdrs = ["activation.h"],
  srcs = ["activation.cc"],
  deps = [
    "//src/common:assert",
    "//src/common:matrix",
  ],
)

cc_library(
  name = "layer",
  hdrs = ["layer.h"],
  srcs = ["layer.cc"],
  deps = [
    ":activation",
    ":cost",
    "//src/common:assert",
    "//src/common:matrix",
  ],
)

cc_library(
  name = "neural_network",
  hdrs = ["neural_network.h"],
  srcs = ["neural_network.cc"],
  deps = [
    ":layer",
    "//src:params",
    "//src/common:assert",
    "//src/common:matrix",
  ],
)

cc_library(
  name = "trainer",
  hdrs = ["trainer.h"],
  srcs = ["trainer.cc"],
  deps = [
    ":neural_network",
    "//src/common:assert",
    "//src/common:matrix",
    "//src/common:thread_pool",
    "//src/io:csv_reader",
  ],
)
