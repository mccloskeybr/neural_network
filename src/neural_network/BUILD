load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
  name = "cost",
  hdrs = ["cost.h"],
  srcs = ["cost.cc"],
  deps = [
    "@abseil-cpp//absl/log:check",
    "@abseil-cpp//absl/status:status",
    "@abseil-cpp//absl/status:statusor",
    "@abseil-cpp//absl/strings:string_view",
    "//src/common:matrix",
  ],
)

cc_library(
  name = "activation",
  hdrs = ["activation.h"],
  srcs = ["activation.cc"],
  deps = [
    "@abseil-cpp//absl/log:check",
    "@abseil-cpp//absl/status:status",
    "@abseil-cpp//absl/status:statusor",
    "@abseil-cpp//absl/strings:string_view",
    "//src/common:matrix",
    "//src/protos:model_checkpoint_cc_proto",
  ],
)

cc_library(
  name = "layer",
  hdrs = ["layer.h"],
  srcs = ["layer.cc"],
  deps = [
    ":activation",
    ":cost",
    ":params",
    "@abseil-cpp//absl/log:check",
    "@abseil-cpp//absl/strings:string_view",
    "//src/common:matrix",
  ],
)

cc_library(
  name = "params",
  hdrs = ["params.h"],
  deps = [
    ":cost",
    "@abseil-cpp//absl/strings:strings",
  ],
)

cc_library(
  name = "neural_network",
  hdrs = ["neural_network.h"],
  srcs = ["neural_network.cc"],
  deps = [
    ":layer",
    ":params",
    "@abseil-cpp//absl/log:check",
    "@abseil-cpp//absl/status:status",
    "@abseil-cpp//absl/status:statusor",
    "@abseil-cpp//absl/strings:strings",
    "//src/common:matrix",
    "//src/protos:model_checkpoint_cc_proto",
  ],
)

cc_library(
  name = "trainer",
  hdrs = ["trainer.h"],
  srcs = ["trainer.cc"],
  deps = [
    ":neural_network",
    ":params",
    "@abseil-cpp//absl/log:check",
    "@abseil-cpp//absl/log:log",
    "@abseil-cpp//absl/strings:strings",
    "@abseil-cpp//absl/status:status",
    "@abseil-cpp//absl/status:statusor",
    "//src/common:matrix",
    "//src/common:thread_pool",
    "//src/io:csv_reader",
    "//src/io:model_checkpoint",
  ],
)
