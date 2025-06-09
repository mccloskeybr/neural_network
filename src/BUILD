load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

cc_library(
  name = "params",
  hdrs = ["params.h"],
  deps = [
    "//src/neural_network:activation",
    "//src/neural_network:cost",
  ],
)

cc_binary(
  name = "main",
  srcs = ["main.cc"],
  deps = [
    "//src/common:assert",
    "//src/common:matrix",
    "//src/io:csv_reader",
    "//src/neural_network:neural_network",
    "//src/neural_network:trainer",
  ],
)
