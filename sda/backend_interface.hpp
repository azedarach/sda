#ifndef SDA_BACKEND_INTERFACE_HPP_INCLUDED
#define SDA_BACKEND_INTERFACE_HPP_INCLUDED

#include <cstddef>

namespace sda {

namespace backends {

namespace detail {

template <class Matrix, class Enable = void>
struct cols_impl {};

template <class Matrix, class Enable = void>
struct rows_impl {};

template <class Scalar, class Matrix, class Enable = void>
struct add_constant_impl {};

template <class Scalar1, class MatrixA, class Scalar2, class MatrixB,
          class MatrixC, class Enable = void>
struct geam_impl {};

template <class Scalar1, class MatrixA, class MatrixB, class Scalar2,
          class MatrixC, class Enable = void>
struct gemm_impl {};

template <class Scalar1, class MatrixA, class MatrixB, class Scalar2,
          class MatrixC, class Enable = void>
struct trace_gemm_impl {};

template <class Scalar1, class MatrixA, class MatrixB,
          class Enable = void>
struct trace_gemm_op_impl {};

template <class Scalar1, class MatrixA, class MatrixB, class Scalar2,
          class MatrixC, class Enable = void>
struct sum_gemm_impl {};

template <class Scalar1, class MatrixA, class MatrixB,
          class Enable = void>
struct sum_gemm_op_impl {};

template <class MatrixC, class MatrixA, class MatrixB, class ResidualMatrix,
          class Enable = void>
struct matrix_residual_impl {};

template <class MatrixC, class MatrixA, class MatrixB,
          class Enable = void>
struct matrix_residual_fro_norm_impl {};

template <class MatrixA, class MatrixB, class Enable = void>
struct solve_qr_impl {};

} // namespace detail

enum class Op_flag {
   None, Transpose, Adjoint
};

/**
 * @brief Returns the number of columns in a matrix.
 *
 * @tparam Matrix the matrix type
 */
template <class Matrix>
std::size_t cols(const Matrix& m)
{
   return detail::cols_impl<Matrix>::get(m);
}

/**
 * @brief Returns the number of rows in a matrix.
 *
 * @tparam Matrix the matrix type
 */
template <class Matrix>
std::size_t rows(const Matrix& m)
{
   return detail::rows_impl<Matrix>::get(m);
}

/**
 * @brief Adds constant value to each entry of a matrix
 *
 * @tparam Scalar the type of the scalar
 * @tparam Matrix the type of the matrix
 */
template <class Scalar, class Matrix>
void add_constant(Scalar s, Matrix& A)
{
   detail::add_constant_impl<Scalar, Matrix>::eval(s, A);
}

/**
 * @brief Computes matrix-matrix addition
 *
 * Computes C = alpha * op(A) + beta * op(B) .
 *
 * @tparam Scalar1 the type of the scalar coefficient alpha
 * @tparam MatrixA the type of the matrix A
 * @tparam Scalar2 the type of the scalar coefficient beta
 * @tparam MatrixB the type of the matrix B
 * @tparam MatrixC the type of the result matrix C
 */
template <class Scalar1, class MatrixA,
          class Scalar2, class MatrixB, class MatrixC>
void geam(Scalar1 alpha, const MatrixA& A, Scalar2 beta, const MatrixB& B,
          MatrixC& C, Op_flag opA = Op_flag::None, Op_flag opB = Op_flag::None)
{
   detail::geam_impl<Scalar1, MatrixA, Scalar2, MatrixB, MatrixC>::eval(
      alpha, A, beta, B, C, opA, opB);
}

/**
 * @brief Computes matrix-matrix multiplication
 *
 * Implements the operation C = alpha * A * B + beta * C .
 * The contents of the matrix C are overwritten.
 *
 * @tparam Scalar1 the type of the scalar coefficient alpha
 * @tparam MatrixA the type of the matrix A
 * @tparam MatrixB the type of the matrix B
 * @tparam Scalar2 the type of the scalar coefficient beta
 * @tparam MatrixC the type of the matrix C
 */
template <class Scalar1, class MatrixA, class MatrixB, class Scalar2,
          class MatrixC>
void gemm(Scalar1 alpha, const MatrixA& A, const MatrixB& B, Scalar2 beta,
          MatrixC& C, Op_flag opA = Op_flag::None, Op_flag opB = Op_flag::None)
{
   detail::gemm_impl<Scalar1, MatrixA, MatrixB, Scalar2, MatrixC>::eval(
      alpha, A, B, beta, C, opA, opB);
}

/**
 * @brief Computes the trace of a matrix-matrix multiplication and addition.
 *
 * Returns the value of Tr[alpha * A * B + beta * C], where Tr
 * denotes a matrix trace.
 *
 * @tparam Scalar1 the type of the scalar coefficient alpha
 * @tparam MatrixA the type of the matrix A
 * @tparam MatrixB the type of the matrix B
 * @tparam Scalar2 the type of the scalar coefficient beta
 * @tparam MatrixC the type of the matrix C
 */
template <class Scalar1, class MatrixA, class MatrixB, class Scalar2,
          class MatrixC>
typename detail::trace_gemm_impl<Scalar1, MatrixA, MatrixB,
  Scalar2, MatrixC>::value_type
trace_gemm(Scalar1 alpha, const MatrixA& A, const MatrixB& B, Scalar2 beta,
           MatrixC& C, Op_flag opA = Op_flag::None,
           Op_flag opB = Op_flag::None)
{
   return detail::trace_gemm_impl<
      Scalar1, MatrixA, MatrixB, Scalar2, MatrixC>::eval(
         alpha, A, B, beta, C, opA, opB);
}

/**
 * @brief Computes the trace of a matrix-matrix multiplication.
 *
 * Returns the value of Tr[alpha * A * B], where Tr
 * denotes a matrix trace.
 *
 * @tparam Scalar1 the type of the scalar coefficient alpha
 * @tparam MatrixA the type of the matrix A
 * @tparam MatrixB the type of the matrix B
 */
template <class Scalar1, class MatrixA, class MatrixB>
typename detail::trace_gemm_op_impl<Scalar1, MatrixA, MatrixB>::value_type
trace_gemm_op(Scalar1 alpha, const MatrixA& A, const MatrixB& B,
              Op_flag opA = Op_flag::None,
              Op_flag opB = Op_flag::None)
{
   return detail::trace_gemm_op_impl<Scalar1, MatrixA, MatrixB>::eval(
      alpha, A, B, opA, opB);
}

/**
 * @brief Computes the element-wise sum of a matrix-matrix multiplication
 *        and addition.
 *
 * Returns the value of \sum_{i, j}(alpha * A * B + beta * C)_{ij}.
 *
 * @tparam Scalar1 the type of the scalar coefficient alpha
 * @tparam MatrixA the type of the matrix A
 * @tparam MatrixB the type of the matrix B
 * @tparam Scalar2 the type of the scalar coefficient beta
 * @tparam MatrixC the type of the matrix C
 */
template <class Scalar1, class MatrixA, class MatrixB, class Scalar2,
          class MatrixC>
typename detail::sum_gemm_impl<Scalar1, MatrixA, MatrixB,
  Scalar2, MatrixC>::value_type
sum_gemm(Scalar1 alpha, const MatrixA& A, const MatrixB& B, Scalar2 beta,
         MatrixC& C, Op_flag opA = Op_flag::None,
         Op_flag opB = Op_flag::None)
{
   return detail::sum_gemm_impl<
      Scalar1, MatrixA, MatrixB, Scalar2, MatrixC>::eval(
         alpha, A, B, beta, C, opA, opB);
}

/**
 * @brief Computes the element-wise sum of a matrix-matrix multiplication.
 *
 * Returns the value of \sum_{i, j}(alpha * A * B)_{ij}.
 *
 * @tparam Scalar1 the type of the scalar coefficient alpha
 * @tparam MatrixA the type of the matrix A
 * @tparam MatrixB the type of the matrix B
 */
template <class Scalar1, class MatrixA, class MatrixB>
typename detail::sum_gemm_op_impl<Scalar1, MatrixA, MatrixB>::value_type
sum_gemm_op(Scalar1 alpha, const MatrixA& A, const MatrixB& B,
            Op_flag opA = Op_flag::None,
            Op_flag opB = Op_flag::None)
{
   return detail::sum_gemm_op_impl<Scalar1, MatrixA, MatrixB>::eval(
      alpha, A, B, opA, opB);
}

/**
 * @brief calculates a matrix residual
 *
 * The residual is defined as r = C - AB .
 *
 * @tparam MatrixC type of the initial matrix C
 * @tparam MatrixA type of the left factor in the matrix product
 * @tparam MatrixB type of the right factor in the matrix product
 * @tparam ResidualMatrix type of the resulting residual
 */
template <class RhsMatrix, class LhsMatrixA, class LhsMatrixB,
          class ResidualMatrix>
void matrix_residual(
   const RhsMatrix& C, const LhsMatrixA& A, const LhsMatrixB& B,
   ResidualMatrix& r)
{
   detail::matrix_residual_impl<
      RhsMatrix, LhsMatrixA, LhsMatrixB, ResidualMatrix>::eval(C, A, B, r);
}

/**
 * @brief calculates the Frobenius norm of a matrix residual
 *
 * The residual is defined as r = C - AB .
 *
 * @tparam MatrixC type of the initial matrix C
 * @tparam MatrixA type of the left factor in the matrix product
 * @tparam MatrixB type of the right factor in the matrix product
 */
template <class MatrixC, class MatrixA, class MatrixB>
typename detail::matrix_residual_fro_norm_impl<MatrixC,
   MatrixA, MatrixB>::value_type
matrix_residual_fro_norm(
   const MatrixC& C, const MatrixA& A, const MatrixB& B)
{
   return detail::matrix_residual_fro_norm_impl<
      MatrixC, MatrixA, MatrixB>::eval(C, A, B);
}

/**
 * @brief Solves a linear system using QR decomposition
 *
 * @tparam MatrixA the type of the coefficients matrix
 * @tparam MatrixB the type of the solution matrix
 */
template <class MatrixA, class MatrixB>
typename detail::solve_qr_impl<MatrixA, MatrixB>::return_type
solve_qr(const MatrixA& A, MatrixB& B)
{
   return detail::solve_qr_impl<MatrixA, MatrixB>::eval(A, B);
}

} // namespace backends

} // namespace sda

#endif
