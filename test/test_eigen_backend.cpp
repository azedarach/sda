#include "catch.hpp"

#include <Eigen/Core>

#include "sda/backends/eigen_backend.hpp"

using namespace sda;

TEST_CASE( "Row and column backend helpers return correct values",
           "[eigen_backend]")
{
   SECTION("Returns correct rows and column for dynamic sized matrices")
   {
      Eigen::MatrixXd x1(2, 2);
      Eigen::MatrixXd x2(Eigen::MatrixXd::Zero(4, 5));

      CHECK(backends::rows(x1) == x1.rows());
      CHECK(backends::cols(x1) == x1.cols());
      CHECK(backends::rows(x2) == x2.rows());
      CHECK(backends::cols(x2) == x2.cols());
   }
}
