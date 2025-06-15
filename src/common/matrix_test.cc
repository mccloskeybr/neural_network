#include "src/common/matrix.h"

#include <gtest/gtest.h>

#include "absl/log/log.h"

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
  double scalar = 2.0f;
  auto a = Matrix(2, 3, {
        1, 2, 3,
        4, 5, 6,
      });
  auto expected = Matrix(2, 3, {
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

  LOG(INFO) << "A:\n" << a.DebugString();
  LOG(INFO) << "B:\n" << b.DebugString();
  LOG(INFO) << "Actual:\n" << (a * b).DebugString();
  LOG(INFO) << "Expected:\n" << expected.DebugString();

  EXPECT_TRUE(a * b == expected);
}

TEST(MatrixTest, TransposeSucceed) {
  auto a = Matrix(2, 3, {
        1, 2, 3,
        4, 5, 6,
      });
  auto expected = Matrix(3, 2, {
        1, 4,
        2, 5,
        3, 6,
      });
  EXPECT_TRUE(a.Transpose() == expected);
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
  auto expected = Matrix(2, 3, {
        7, 16, 27,
        44, 60, 78,
      });
  EXPECT_TRUE(a.HadamardMult(b) == expected);
}

TEST(MatrixTest, HadamardMultiplyInPlaceSucceed) {
  auto a = Matrix(2, 3, {
        1, 2, 3,
        4, 5, 6,
      });
  auto b = Matrix(2, 3, {
        7, 8, 9,
        11, 12, 13,
      });
  auto expected = Matrix(2, 3, {
        7, 16, 27,
        44, 60, 78,
      });
  a.HadamardMultInPlace(b);
  EXPECT_TRUE(a == expected);
}

TEST(MatrixTest, ClassifySucceed) {
  auto x = Matrix(1, 5, {0, -1, 0, 1, 0.5 });
  EXPECT_TRUE(x.Classify() == 3);
}
