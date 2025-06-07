#include "matrix.h"

#include <gtest/gtest.h>

TEST(MatrixTest, AddSucceed) {
  auto a = Matrix<2, 2>({
        1, 1,
        1, 1,
      });
  auto b = Matrix<2, 2>({
        2, 2,
        2, 2,
      });
  auto expected = Matrix<2, 2>({
        3, 3,
        3, 3,
      });
  EXPECT_TRUE(a + b == expected);
}

TEST(MatrixTest, MultiplySucceed) {
  auto a = Matrix<2, 3>({
        1, 2, 3,
        4, 5, 6,
      });
  auto b = Matrix<3, 4>({
        7, 8, 9, 10,
        11, 12, 13, 14,
        15, 16, 17, 18,
      });
  auto expected = Matrix<2, 4>({
        74, 80, 86, 92,
        173, 188, 203, 218,
      });
  EXPECT_TRUE(a * b == expected);
}
