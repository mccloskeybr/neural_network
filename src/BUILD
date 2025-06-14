load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

cc_binary(
  name = "main",
  srcs = ["main.cc"],
  deps = [
    "@abseil-cpp//absl/flags:flag",
    "@abseil-cpp//absl/flags:parse",
    "@abseil-cpp//absl/log:check",
    "@abseil-cpp//absl/log:initialize",
    "@abseil-cpp//absl/log:log",
    "@abseil-cpp//absl/log:flags",
    "@abseil-cpp//absl/status:statusor",
    "@abseil-cpp//absl/strings:strings",
    "//src/common:assert",
    "//src/common:matrix",
    "//src/io:csv_reader",
    "//src/io:model_checkpoint",
    "//src/neural_network:neural_network",
    "//src/neural_network:trainer",
    "//src/neural_network:params",
  ],
)
