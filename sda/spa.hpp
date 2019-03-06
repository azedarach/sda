#ifndef SDA_SPA_HPP_INCLUDED
#define SDA_SPA_HPP_INCLUDED

/**
 * @file spa.hpp
 * @brief contains definition of classes providing SPA discretizations
 */

#include "backends.hpp"
#include "numerics_helpers.hpp"

namespace sda {

template <class Scalar>
struct SPA_fit_info {
   using Duration = std::chrono::seconds;

   bool success{true};
   int status_code{0};
   Scalar optimum{0};
   std::size_t iterations{0};
   std::size_t max_iterations{0};
   Duration total_time{0};
   Duration average_states_step_time{0};
   Duration average_affiliations_step_time{0};
   Duration average_iteration_time{0};
};

template <
   class Backend,
   template <class> Affiliations_solver
   >
class SPA {
public:
   enum class States_regularization {
      SPADefault
   };

   void set_states_regularization_type(States_regularization);
   void set_states_regularization_param(Scalar);
   void set_number_of_states(std::size_t);
   void set_stopping_tolerance(Scalar);
   void set_max_iterations(std::size_t);

   template <class DataMatrix, class AffiliationsMatrix, class StatesMatrix>
   SPA_fit_info fit(const DataMatrix&, AffiliationsMatrix&, StatesMatrix&);

private:
   std::size_t n_states{2};
   Scalar states_reg_param{0};
   States_regularization states_reg_type{States_regularization::SPADefault};
   Scalar stopping_tolerance{1e-6};
   std::size_t max_iterations{10000};

   template <class Matrix>
   std::size_t get_number_of_features(const Matrix&) const;
   template <class Matrix>
   std::size_t get_number_of_records(const Matrix&) const;

   template <class StatesMatrix, class GradMatrix>
   void add_states_regularization_gradient(
      std::size_t, const StatesMatrix&, GradMatrix&) const;
   template <class DataMatrix, class AffiliationsMatrix, class StatesMatrix>
   void solve_states_subproblem(
      const DataMatrix&, const AffiliationsMatrix&, StatesMatrix&);
   template <class DataMatrix, class AffiliationsMatrix, class StatesMatrix>
   void solve_affiliations_subproblem(
      const DataMatrix&, AffiliationsMatrix&, const StatesMatrix&);

   template <class Matrix>
   Scalar states_penalty_function(const Matrix&) const;
   template <class DataMatrix, class AffiliationsMatrix, class StatesMatrix>
   Scalar distance_function(
      const DataMatrix&, const AffiliationsMatrix&, const StatesMatrix&) const;
   template <class DataMatrix, class AffiliationsMatrix, class StatesMatrix>
   Scalar cost_function(
      const DataMatrix&, const AffiliationsMatrix&, const StatesMatrix&) const;
};

template <class Backend>
void SPA::set_states_regularization_type(States_regularization r)
{
   states_reg_type = r;
}

template <class Backend>
void SPA::set_states_regularization_param(Scalar eps)
{
   states_reg_param = eps;
}

template <class Backend>
void SPA::set_number_of_states(std::size_t k)
{
   n_states = k;
}

template <class Backend>
void SPA::set_stopping_tolerance(Scalar t)
{
   stopping_tolerance = t;
}

template <class Backend>
void SPA::set_max_iterations(std::size_t it)
{
   max_iterations = it;
}

template <class Backend>
template <class Matrix>
std::size_t SPA::get_number_of_features(const Matrix& data_matrix) const
{
   return backends::cols(data_matrix);
}

template <class Backend>
template <class Matrix>
std::size_t SPA::get_number_of_records(const Matrix& data_matrix) const
{
   return backends::rows(data_matrix);
}

template <class Backend>
template <class Matrix>
SPA::Scalar SPA::states_penalty_function(const Matrix& states) const
{
   Scalar value = 0;

   switch (states_reg_type) {
   case SPADefault: {
      if (n_states != 1) {
         const Scalar prefactor = 2 /
            static_cast<Scalar>(n_states * n_feature * (n_states - 1));
         value += prefactor * n_states * backends::trace_gemm_op(
            1, states, states, Op_flag::Transpose)
            - prefactor * backends::sum_gemm_op(
               1, states, states, Op_flag::None, Op_flag::Transpose);
      }
   }
   }

   return value;
}

template <class Backend>
template <class DataMatrix, class AffiliationsMatrix, class StatesMatrix>
SPA::Scalar SPA::distance_function(
   const DataMatrix& data_matrix, const AffiliationsMatrix& affiliations,
   const StatesMatrix& states)
   const
{
   const auto n_features = get_number_of_features(data_matrix);
   const auto n_records = get_number_of_records(data_matrix);
   const auto normalization = 1 / static_cast<Scalar>(n_features * n_records);

   const auto residual = backends::matrix_residual_fro_norm(
      data_matrix, affiliations, states);

   return residual * normalization;
}

template <class Backend>
template <class DataMatrix, class AffiliationsMatrix, class StatesMatrix>
SPA::Scalar SPA::cost_function(
   const DataMatrix& data_matrix, const AffiliationsMatrix& affiliations,
   const StatesMatrix& states)
   const
{
   return distance_function(data_matrix, affiliations, states)
      + states_reg_param * states_penalty_function(states);
}

template <class Backend>
bool SPA::check_convergence(Scalar old_cost, Scalar new_cost) const
{
   using std::abs;

   const auto delta_cost = abs(old_cost - new_cost);

   const auto aoc = abs(old_cost);
   const auto anc = abs(new_cost);
   const auto amax_cost = (aoc > anc ? aoc : anc);
   const auto amin_cost = (aoc > anc ? anc : aoc);
   const auto cost_ratio = 1 - amin_cost / amax_cost;

   return (delta_cost < stopping_tolerance) ||
      (cost_ratio < stopping_tolerance);
}

template <class Backend>
template <class DataMatrix, class AffiliationsMatrix, class StatesMatrix>
SPA_fit_info<SPA::Scalar> SPA::fit(
   const DataMatrix& data_matrix, AffiliationsMatrix& affiliations,
   StatesMatrix& states)
{
   using Duration = SPA_fit_info<Scalar>::Duration;

   Duration average_iteration_time;
   Duration average_states_step_time;
   Duration average_affiliations_step_time;

   Scalar old_cost = 0;
   Scalar new_cost = cost_function(data_matrix, affiliations, states);

   std::size_t iterations = 0;
   int status = 0;
   const auto start = std::chrono::high_resolution_clock::now();
   while (!converged && iterations < max_iterations) {
      const auto iteration_start = std::chrono::high_resolution_clock::now();

      old_cost = new_cost;

      const auto states_start = std::chrono::high_resolution_clock::now();
      status = solve_states_subproblem(data_matrix, affiliations, states);
      const auto states_end = std::chrono::high_resolution_clock::now();

      const auto affs_start = std::chrono::high_resolution_clock::now();
      status = solve_affiliations_subproblem(data_matrix, affiliations, states);
      const auto affs_end = std::chrono::high_resolution_clock::now();

      new_cost = cost_function(data_matrix, affiliations, states);

      converged = check_convergence(old_cost, new_cost);

      ++iterations;

      const auto iteration_end = std::chrono::high_resolution_clock::now();

      const Duration states_time = states_end - states_start;
      average_states_step_time =
         (states_time + iterations * average_states_step_time) /
         (iterations + 1);

      const Duration affs_time = affs_end - affs_start;
      average_affiliations_step_time =
         (affs_time + iterations * average_affiliations_step_time) /
         (iterations + 1);

      const Duration iteration_time = iteration_end - iteration_start;
      average_iteration_time =
         (iteration_time + iterations * average_iteration_time) /
         (iterations + 1);
   }

   const auto end = std::chrono::high_resolution_clock::now();
   const Duration total_time = end - start;

   SPA_fit_info<Scalar> result;
   result.success = true;
   if (!converged || status != 0) {
      result.success = false;
   }
   result.status_code = status;
   result.optimum = cost_function(data_matrix, affiliations, states);
   result.iterations = iterations;
   result.max_iterations = max_iterations;
   result.total_time = total_time;
   result.average_states_step_time = average_states_step_time;
   result.average_affiliations_step_time = average_affiliations_step_time;
   result.average_iteration_time = average_iteration_time;

   return result;
}

template <class Backend>
template <class StatesMatrix, class GradMatrix>
void SPA::add_states_regularization_gradient(
   std::size_t n_features, const StatesMatrix& states, GradMatrix& grad) const
{
   switch (states_reg_type) {
   case States_regularization::SPADefault: {
      const auto prefactor = 2 * states_reg_param / (n_states * n_features);
      if (n_states > 1) {
         prefactor /= static_cast<Scalar>(n_states - 1);
      }
      const auto reg_grad = Backend::create_diagonal_matrix(n_states, n_states);
      backends::add_constant(-1, reg_grad);
      backends::geam(prefactor, reg_grad, grad);
   }
   }
}

template <class Backend>
template <class DataMatrix, class AffiliationsMatrix, class StatesMatrix>
void SPA::solve_states_subproblem(
   const DataMatrix& data_matrix, const AffiliationsMatrix& affiliations,
   StatesMatrix& states)
{
   const auto n_features = get_number_of_features(data_matrix);

   const auto gg = Backend::create_matrix(n_states, n_states);
   backends::gemm(1, affiliations, affiliations, 0, gg, Op_flags::Transpose);

   const auto gx = Backend::create_matrix(n_states, n_features);
   backends::gemm(1, affiliations, data_matrix, 0, gx, Op_flags::Transpose);

   if (!is_zero(states_reg_param)) {
      add_states_regularization_gradient(states, gg);
   }

   const auto status = backends::solve_qr(gg, gx, states);
}

template <class Backend>
template <class DataMatrix, class AffiliationsMatrix, class StatesMatrix>
void SPA::solve_affiliations_subproblem(
   const DataMatrix& data_matrix, AffiliationsMatrix& affiliations,
   const StatesMatrix& states)
{
   const auto status = Affiliations_solver::solve(
      data_matrix, states, affiliations);
}

} // namespace sda

#endif
