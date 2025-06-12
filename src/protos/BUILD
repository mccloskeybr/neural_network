load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")

package(default_visibility = ["//visibility:public"])

proto_library(
  name = "model_checkpoint_proto",
  srcs = ["model_checkpoint.proto"],
  deps = [],
)

cc_proto_library(
  name = "model_checkpoint_cc_proto",
  deps = [":model_checkpoint_proto"],
)
