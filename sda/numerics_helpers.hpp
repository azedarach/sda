#ifndef SDA_NUMERICS_HELPERS_HPP_INCLUDED
#define SDA_NUMERICS_HELPERS_HPP_INCLUDED

#include <numeric_limits>
#include <type_traits>

namespace sda {

template <typename T>
typename std::enable_if<std::is_unsigned<T>::value, bool>::type
is_zero(T a, T tol = std::numeric_limits<T>::epsilon)
{
   return a < tol;
}

template <typename T>
typename std::enable_if<!std::is_unsigned<T>::value &&
                        std::is_arithmetic<T>::value, bool>::type
is_zero(T a, T tol = std::numeric_limits<T>::epsilon())
{
   using std::abs;

   return abs(a) < tol;
}

} // namespace sda

#endif
