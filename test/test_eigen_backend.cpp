#include "catch.hpp"

#include <Eigen/Core>

#include "sda/backends/eigen_backend.hpp"

#include "comparison_helpers.hpp"

using namespace sda;

TEST_CASE( "Row and column backend helpers return correct values",
           "[eigen_backend]")
{
   SECTION("Returns correct rows and columns for dynamic sized matrices")
   {
      Eigen::MatrixXd x1(2, 2);
      Eigen::MatrixXd x2(Eigen::MatrixXd::Zero(4, 5));

      CHECK(backends::rows(x1) == x1.rows());
      CHECK(backends::cols(x1) == x1.cols());
      CHECK(backends::rows(x2) == x2.rows());
      CHECK(backends::cols(x2) == x2.cols());
   }

   SECTION("Returns correct rows and columns for fixed size matrices")
   {
      Eigen::Matrix<double, 3, 6> x1;
      Eigen::Matrix<int, 2, 1> x2;

      CHECK(backends::rows(x1) == x1.rows());
      CHECK(backends::cols(x1) == x1.cols());
      CHECK(backends::rows(x2) == x2.rows());
      CHECK(backends::cols(x2) == x2.cols());
   }
}

TEST_CASE( "Residual matrix and norm correctly computed",
           "[eigen_backend]")
{
   SECTION("Returns correct residual for dynamic sized matrices")
   {
      Eigen::MatrixXd C(Eigen::MatrixXd::Random(4, 4));
      Eigen::MatrixXd A(Eigen::MatrixXd::Random(4, 3));
      Eigen::MatrixXd B(Eigen::MatrixXd::Random(3, 4));
      Eigen::MatrixXd residual(4, 4);

      backends::matrix_residual(C, A, B, residual);

      Eigen::MatrixXd expected = C - A * B;

      CHECK(is_equal(residual, expected));
   }
}
