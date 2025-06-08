load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
  name = "config",
  hdrs = ["config.h"],
  deps = [
    ":activation",
    ":cost",
  ],
)

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
  ],
)

cc_library(
  name = "layer",
  hdrs = ["layer.h"],
  srcs = ["layer.cc"],
  deps = [
    ":activation",
    ":config",
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
    ":config",
    ":layer",
    "//src/common:assert",
    "//src/common:matrix",
  ],
)

