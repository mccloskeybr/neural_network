load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
  name = "csv_reader",
  hdrs = ["csv_reader.h"],
  srcs = ["csv_reader.cc"],
  deps = [
    "//src/common:matrix",
  ],
)

cc_library(
  name = "model_checkpoint",
  hdrs = ["model_checkpoint.h"],
  srcs = ["model_checkpoint.cc"],
  deps = [
    "//src/common:assert",
    "//src/protos:model_checkpoint_cc_proto",
  ],
)
