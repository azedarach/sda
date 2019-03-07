#ifndef SDA_SPA_AFFILIATIONS_SOLVER_POLICY_HPP_INCLUDED
#define SDA_SPA_AFFILIATIONS_SOLVER_POLICY_HPP_INCLUDED

/**
 * @file spa_affiliations_solver_policy.hpp
 * @brief contains implementations of affiliation solver policies
 */

namespace sda {

template <class Backend>
struct Descent_step_solver {
   using return_type = int;

   template <class DataMatrix, class AffiliationsMatrix, class StatesMatrix>
   static return_type solve(const DataMatrix&, AffiliationsMatrix&,
                            const StatesMatrix&);
};

template <class Backend>
template <class DataMatrix, class AffiliationsMatrix, class StatesMatrix>
typename Descent_step_solver<Backend>::return_type
Descent_step_solver<Backend>::solve(
   const DataMatrix&, AffiliationsMatrix&, const StatesMatrix&)
{
   const auto xs = Backend::create_matrix(n_records, n_clusters);
   backends::gemm(-2, data_matrix, states, Op_flag::None, Op_flag::Transpose);

   const auto ss = Backend::create_matrix(n_states, n_states);
   backends::gemm(2, states, states, Op_flag::None, Op_flag::Transpose);

   simplex_projection(affiliations);
}

} // namespace sda

#endif
