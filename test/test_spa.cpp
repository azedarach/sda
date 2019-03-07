#include "catch.hpp"

#include "sda/spa.hpp"

using namespace sda;

#ifdef HAVE_EIGEN

#include "sda/backends/eigen_backend.hpp"

TEST_CASE( "State regularization parameters correctly set with Eigen backend",
           "[spa][eigen_backend]")
{
   SECTION("Require states regularization parameter is non-negative")
   {
      using backend_type = backends::Eigen_backend<double>;
      EuclideanSPA<backend_type> model;
      CHECK_THROWS(model.set_states_regularization_param(-1));
   }
}

#endif
