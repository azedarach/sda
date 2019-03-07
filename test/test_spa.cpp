#include "catch.hpp"

#include "sda/spa.hpp"

using namespace sda;

TEST_CASE( "Test state regularization parameters correctly set",
           "[spa]")
{
   SECTION("Require states regularization parameter is non-negative")
   {
      EuclideanSPA model;
      CHECK_THROWS(model.set_states_regularization_param(-1));
   }
}
