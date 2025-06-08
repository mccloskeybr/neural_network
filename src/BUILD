cc_library(
  name = "matrix",
  hdrs = ["matrix.h"],
  srcs = ["matrix.cc"],
  deps = [],
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
  deps = [],
)

cc_library(
  name = "activation",
  hdrs = ["activation.h"],
  srcs = ["activation.cc"],
  deps = [],
)

cc_library(
  name = "layer",
  hdrs = ["layer.h"],
  srcs = ["layer.cc"],
  deps = [
    ":activation",
    ":config",
    ":cost",
    ":matrix",
  ],
)

cc_library(
  name = "neural_network",
  hdrs = ["neural_network.h"],
  srcs = ["neural_network.cc"],
  deps = [
    ":config",
    ":layer",
    ":matrix",
  ],
)

cc_binary(
  name = "main",
  srcs = ["main.cc"],
  deps = [
    ":matrix",
    ":neural_network",
  ],
)
