#include "catch.hpp"

#include <Eigen/Core>

#include "sda/backends/eigen_backend.hpp"

#include "comparison_helpers.hpp"
#include "sda/numerics_helpers.hpp"

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

TEST_CASE( "Residual matrix correctly computed",
           "[eigen_backend]")
{
   SECTION("Returns correct residual for dynamic sized matrices")
   {
      Eigen::MatrixXd C(Eigen::MatrixXd::Random(4, 4));
      Eigen::MatrixXd A(Eigen::MatrixXd::Random(4, 3));
      Eigen::MatrixXd B(Eigen::MatrixXd::Random(3, 4));
      Eigen::MatrixXd r1(4, 4);

      backends::matrix_residual(C, A, B, r1);

      Eigen::MatrixXd expected = C - A * B;

      CHECK(is_equal(r1, expected));

      A = Eigen::MatrixXd::Random(10, 10);
      B = Eigen::MatrixXd::Random(12, 10);
      Eigen::MatrixXd r2(10, 12);

      backends::matrix_residual(A * B, A, B, r2);

      CHECK(is_equal(r2, Eigen::MatrixXd::Zero(10, 12)));
   }

   SECTION("Returns correct residual for fixed size matrices")
   {
      Eigen::Matrix<double, 3, 3> C(Eigen::MatrixXd::Random(3, 3));
      Eigen::Matrix<double, 3, 5> A(Eigen::MatrixXd::Random(3, 5));
      Eigen::Matrix<double, 5, 3> B(Eigen::MatrixXd::Random(5, 3));
      Eigen::Matrix<double, 3, 3> r1;

      backends::matrix_residual(C, A, B, r1);
      Eigen::Matrix<double, 3, 3> exp1 = C - A * B;

      CHECK(is_equal(r1, exp1));
   }
}

TEST_CASE( "Residual matrix norm correctly computed",
           "[eigen_backend]")
{
   SECTION("Returns correct norm for dynamic size matrices")
   {
      Eigen::MatrixXd C(Eigen::MatrixXd::Random(6, 6));
      Eigen::MatrixXd A(Eigen::MatrixXd::Random(6, 6));
      Eigen::MatrixXd B(Eigen::MatrixXd::Random(6, 6));

      const auto rn1 = backends::matrix_residual_fro_norm(
         C, A, B);
      const double exp1 = (C - A * B).norm();

      CHECK(is_equal(rn1, exp1));
   }

   SECTION("Returns correct norm for fixed size matrices")
   {
      Eigen::Matrix<double, 4, 4> C(Eigen::MatrixXd::Random(4, 4));
      Eigen::Matrix<double, 4, 5> A(Eigen::MatrixXd::Random(4, 5));
      Eigen::Matrix<double, 5, 4> B(Eigen::MatrixXd::Random(5, 4));

      const auto rn1 = backends::matrix_residual_fro_norm(
         C, A, B);
      const double exp1 = (C - A * B).norm();

      CHECK(is_equal(rn1, exp1));
   }
}
