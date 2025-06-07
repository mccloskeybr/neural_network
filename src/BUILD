cc_library(
  name = "matrix",
  hdrs = ["matrix.h"],
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

cc_binary(
  name = "main",
  srcs = ["main.cc"],
  deps = [
    ":matrix",
  ],
)
