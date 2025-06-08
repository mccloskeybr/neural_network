load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

cc_binary(
  name = "main",
  srcs = ["main.cc"],
  deps = [
    "//src/common:assert",
    "//src/common:matrix",
    "//src/io:csv_reader",
    "//src/neural_network:neural_network",
  ],
)
