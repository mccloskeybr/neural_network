#include "src/common/matrix.h"

#include <gtest/gtest.h>

TEST(MatrixTest, AddSucceed) {
  auto a = Matrix(2, 2, {
        1, 1,
        1, 1,
      });
  auto b = Matrix(2, 2, {
        2, 2,
        2, 2,
      });
  auto expected = Matrix(2, 2, {
        3, 3,
        3, 3,
      });
  EXPECT_TRUE(a + b == expected);
}

TEST(MatrixTest, ScalarMultiplySucceed) {
  float scalar = 2.0f;
  auto a = Matrix(2, 3, {
        1, 2, 3,
        4, 5, 6,
      });
  auto expected = Matrix(2, 4, {
        2, 4, 6,
        8, 10, 12,
      });
  EXPECT_TRUE(a * scalar == expected);
}

TEST(MatrixTest, MatMultiplySucceed) {
  auto a = Matrix(2, 3, {
        1, 2, 3,
        4, 5, 6,
      });
  auto b = Matrix(3, 4, {
        7, 8, 9, 10,
        11, 12, 13, 14,
        15, 16, 17, 18,
      });
  auto expected = Matrix(2, 4, {
        74, 80, 86, 92,
        173, 188, 203, 218,
      });
  EXPECT_TRUE(a * b == expected);
}

TEST(MatrixTest, HadamardMultiplySucceed) {
  auto a = Matrix(2, 3, {
        1, 2, 3,
        4, 5, 6,
      });
  auto b = Matrix(2, 3, {
        7, 8, 9,
        11, 12, 13,
      });
  auto expected = Matrix(2, 4, {
        7, 16, 27,
        44, 60, 78,
      });
  EXPECT_TRUE(a.HadamardMult(b) == expected);
}

TEST(MatrixTest, TransposeSucceed) {
  auto x = Matrix(2, 3, {
        1, 2, 3,
        4, 5, 6,
      });
  auto expected = Matrix(3, 2, {
      1, 4,
      2, 5,
      3, 6,
      });
  EXPECT_TRUE(x.Transpose() == expected);
}

TEST(MatrixTest, MapSucceed) {
  auto x = Matrix(2, 3, {
        1, 2, 3,
        4, 5, 6,
      });
  auto expected = Matrix(2, 3, {
        10, 20, 30,
        40, 50, 60,
      });
  EXPECT_TRUE(x.Map([](float x) { return x * 10; }) == expected);
}

TEST(MatrixTest, MergeSucceed) {
  auto x = Matrix(2, 2, {
        1, 2,
        3, 4,
      });
  auto y = Matrix(2, 2, {
        10, 20,
        30, 40,
      });
  auto expected = Matrix(2, 2, {
        11, 22,
        33, 44,
      });
  EXPECT_TRUE(x.Merge(y, [](float x, float y) { return x + y; }) == expected);
}

TEST(MatrixTest, ClassifySucceed) {
  auto x = Matrix(1, 5, {0, -1, 0, 1, 0.5 });
  EXPECT_TRUE(x.Classify() == 3);
}
