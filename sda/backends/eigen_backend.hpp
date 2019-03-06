#ifndef SDA_EIGEN_BACKEND_HPP_INCLUDED
#define SDA_EIGEN_BACKEND_HPP_INCLUDED

#include <Eigen/Core>

#include <type_traits>

/**
 * @file eigen_backend.hpp
 * @brief provides implementation of Eigen3 backend
 */

namespace sda {

namespace backends {

template <class Scalar>
struct Eigen_backend {
   using Index = Eigen::Index;

   static Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
   create_matrix(Index rows, Index cols)
      {
         return Eigen::Matrix<
            Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(rows, cols);
      }

   static Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
   create_constant_matrix(Index rows, Index cols, Scalar value)
      {
         return Eigen::Matrix<
            Scalar, Eigen::Dynamic, Eigen::Dynamic>::Constant(
               rows, cols, value);
      }

   static Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>
   create_diagonal_matrix(Index size, Scalar diagonal = Scalar(1));
      {
         return diagonal * Eigen::Matrix<
            Scalar, Eigen::Dynamic, Eigen::Dynamic>::Identity(rows, cols);
      }
};

namespace detail {

template <class T, class Enable = void>
struct is_eigen_matrix : public std::false_type {};

template <class Matrix>
struct is_eigen_matrix<
   Matrix,
   typename std::enable_if<
      std::is_base_of<Eigen::MatrixBase<Matrix>,
                      Matrix> >::type>
   : public std::true_type {}

template <class Matrix>
struct rows_impl<
   Matrix,
   typename std::enable_if<is_eigen_matrix<Matrix>::value>::type> {

   static std::size_t get(const Matrix& m)
      {
         return static_cast<std::size_t>(m.rows());
      }
};

template <class Matrix>
struct cols_impl<
   Matrix,
   typename std::enable_if<is_eigen_matrix<Matrix>::value>::type> {

   static std::size_t get(const Matrix& m)
      {
         return static_cast<std::size_t>(m.cols());
      }
};

template <class Scalar, class Matrix>
struct add_constant_impl<
   Scalar, Matrix,
   typename std::enable_if<is_eigen_matrix<Matrix>::value>::type> {

   static void eval(Scalar c, Matrix& A)
      {
         A.colwise() += Eigen::VectorXd::Constant(A.rows(), c);
      }
};

template <class Scalar1, class MatrixA, class Scalar2, class MatrixB,
          class MatrixC>
struct geam_impl<
   Scalar1, MatrixA, Scalar2, MatrixB, MatrixC,
   typename std::enable_if<is_eigen_matrix<MatrixA>::value &&
                           is_eigen_matrix<MatrixB>::value &&
                           is_eigen_matrix<MatrixC>::value>::type> {

   static void eval(Scalar1 alpha, const MatrixA& A, Scalar2 beta,
                    const MatrixB& B, MatrixC& C,
                    Op_flag opA, Op_flag opB)
      {
         if (opA == Op_flag::None && opB == Op_flag::None) {
            C = alpha * A + beta * B;
         } else if (opA == Op_flag::Transpose && opB == Op_flag::None) {
            C = alpha * A.transpose() + beta * B;
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::None) {
            C = alpha * A.adjoint() + beta * B;
         } else if (opA == Op_flag::None && opB == Op_flag::Transpose) {
            C = alpha * A + beta * B.transpose();
         } else if (opA == Op_flag::Transpose && opB == Op_flag::Transpose) {
            C = alpha * A.transpose() + beta * B.transpose();
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::Transpose) {
            C = alpha * A.adjoint() + beta * B.transpose();
         } else if (opA == Op_flag::None && opB == Op_flag::Adjoint) {
            C = alpha * A + beta * B.adjoint();
         } else if (opA == Op_flag::Transpose && opB == Op_flag::Adjoint) {
            C = alpha * A.transpose() + beta * B.adjoint();
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::Adjoint) {
            C = alpha * A.adjoint() + beta * B.adjoint();
         }
      }
};

template <class Scalar1, class MatrixA, class MatrixB, class Scalar2,
          class MatrixC>
struct gemm_impl<
   Scalar1, MatrixA, MatrixB, Scalar2, MatrixC,
   typename std::enable_if<is_eigen_matrix<MatrixA>::value &&
                           is_eigen_matrix<MatrixB>::value &&
                           is_eigen_matrix<MatrixC>::value>::type> {

   static void eval(Scalar1 alpha, const MatrixA& A, const MatrixB& B,
                    Scalar2 beta, MatrixC& C, Op_flag opA, Op_flag opB)
      {
         if (opA == Op_flag::None && opB == Op_flag::None) {
            C = alpha * A * B + beta * C;
         } else if (opA == Op_flag::Transpose && opB == Op_flag::None) {
            C = alpha * A.transpose() * B + beta * C;
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::None) {
            C = alpha * A.adjoint() * B + beta * C;
         } else if (opA == Op_flag::None && opB == Op_flag::Transpose) {
            C = alpha * A * B.transpose() + beta * C;
         } else if (opA == Op_flag::Transpose && opB == Op_flag::Transpose) {
            C = alpha * A.transpose() * B.transpose() + beta * C;
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::Transpose) {
            C = alpha * A.adjoint() * B.transpose() + beta * C;
         } else if (opA == Op_flag::None && opB == Op_flag::Adjoint) {
            C = alpha * A * B.adjoint() + beta * C;
         } else if (opA == Op_flag::Transpose && opB == Op_flag::Adjoint) {
            C = alpha * A.transpose() * B.adjoint() + beta * C;
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::Adjoint) {
            C = alpha * A.adjoint() * B.adjoint() + beta * C;
         }
      }
};

template <class Scalar1, class MatrixA, class MatrixB, class Scalar2,
          class MatrixC>
struct trace_gemm_impl<
   Scalar1, MatrixA, MatrixB, Scalar2, MatrixC,
   typename std::enable_if<is_eigen_matrix<MatrixA>::value &&
                           is_eigen_matrix<MatrixB>::value &&
                           is_eigen_matrix<MatrixC>::value>::type> {

   using value_type = typename std::common_type<
      Scalar1, MatrixA::Scalar, MatrixA::Scalar,
      Scalar2, MatrixC::Scalar>::type;

   static value_type eval(Scalar1 alpha, const MatrixA& A, const MatrixB& B,
                          Scalar2 beta, MatrixC& C, Op_flag opA, Op_flag opB)
      {
         if (opA == Op_flag::None && opB == Op_flag::None) {
            return (alpha * A * B + beta * C).trace();
         } else if (opA == Op_flag::Transpose && opB == Op_flag::None) {
            return (alpha * A.transpose() * B + beta * C).trace();
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::None) {
            return (alpha * A.adjoint() * B + beta * C).trace();
         } else if (opA == Op_flag::None && opB == Op_flag::Transpose) {
            return (alpha * A * B.transpose() + beta * C).trace();
         } else if (opA == Op_flag::Transpose && opB == Op_flag::Transpose) {
            return (alpha * A.transpose() * B.transpose() + beta * C).trace();
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::Transpose) {
            return (alpha * A.adjoint() * B.transpose() + beta * C).trace();
         } else if (opA == Op_flag::None && opB == Op_flag::Adjoint) {
            return (alpha * A * B.adjoint() + beta * C).trace();
         } else if (opA == Op_flag::Transpose && opB == Op_flag::Adjoint) {
            return (alpha * A.transpose() * B.adjoint() + beta * C).trace();
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::Adjoint) {
            return (alpha * A.adjoint() * B.adjoint() + beta * C).trace();
         }
      }
};

template <class Scalar1, class MatrixA, class MatrixB>
struct trace_gemm_op_impl<
   Scalar1, MatrixA, MatrixB, Scalar2,
   typename std::enable_if<is_eigen_matrix<MatrixA>::value &&
                           is_eigen_matrix<MatrixB>::value>::type> {

   using value_type = typename std::common_type<
      Scalar1, MatrixA::Scalar, MatrixA::Scalar>::type;

   static value_type eval(Scalar1 alpha, const MatrixA& A, const MatrixB& B,
                          Op_flag opA, Op_flag opB)
      {
         if (opA == Op_flag::None && opB == Op_flag::None) {
            return (alpha * A * B).trace();
         } else if (opA == Op_flag::Transpose && opB == Op_flag::None) {
            return (alpha * A.transpose() * B).trace();
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::None) {
            return (alpha * A.adjoint() * B).trace();
         } else if (opA == Op_flag::None && opB == Op_flag::Transpose) {
            return (alpha * A * B.transpose()).trace();
         } else if (opA == Op_flag::Transpose && opB == Op_flag::Transpose) {
            return (alpha * A.transpose() * B.transpose()).trace();
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::Transpose) {
            return (alpha * A.adjoint() * B.transpose()).trace();
         } else if (opA == Op_flag::None && opB == Op_flag::Adjoint) {
            return (alpha * A * B.adjoint()).trace();
         } else if (opA == Op_flag::Transpose && opB == Op_flag::Adjoint) {
            return (alpha * A.transpose() * B.adjoint()).trace();
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::Adjoint) {
            return (alpha * A.adjoint() * B.adjoint()).trace();
         }
      }
};

template <class Scalar1, class MatrixA, class MatrixB, class Scalar2,
          class MatrixC>
struct sum_gemm_impl<
   Scalar1, MatrixA, MatrixB, Scalar2, MatrixC,
   typename std::enable_if<is_eigen_matrix<MatrixA>::value &&
                           is_eigen_matrix<MatrixB>::value &&
                           is_eigen_matrix<MatrixC>::value>::type> {

   using value_type = typename std::common_type<
      Scalar1, MatrixA::Scalar, MatrixA::Scalar,
      Scalar2, MatrixC::Scalar>::type;

   static value_type eval(Scalar1 alpha, const MatrixA& A, const MatrixB& B,
                          Scalar2 beta, MatrixC& C, Op_flag opA, Op_flag opB)
      {
         if (opA == Op_flag::None && opB == Op_flag::None) {
            return (alpha * A * B + beta * C).sum();
         } else if (opA == Op_flag::Transpose && opB == Op_flag::None) {
            return (alpha * A.transpose() * B + beta * C).sum();
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::None) {
            return (alpha * A.adjoint() * B + beta * C).sum();
         } else if (opA == Op_flag::None && opB == Op_flag::Transpose) {
            return (alpha * A * B.transpose() + beta * C).sum();
         } else if (opA == Op_flag::Transpose && opB == Op_flag::Transpose) {
            return (alpha * A.transpose() * B.transpose() + beta * C).sum();
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::Transpose) {
            return (alpha * A.adjoint() * B.transpose() + beta * C).sum();
         } else if (opA == Op_flag::None && opB == Op_flag::Adjoint) {
            return (alpha * A * B.adjoint() + beta * C).sum();
         } else if (opA == Op_flag::Transpose && opB == Op_flag::Adjoint) {
            return (alpha * A.transpose() * B.adjoint() + beta * C).sum();
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::Adjoint) {
            return (alpha * A.adjoint() * B.adjoint() + beta * C).sum();
         }
      }
};

template <class Scalar1, class MatrixA, class MatrixB>
struct sum_gemm_op_impl<
   Scalar1, MatrixA, MatrixB, Scalar2,
   typename std::enable_if<is_eigen_matrix<MatrixA>::value &&
                           is_eigen_matrix<MatrixB>::value>::type> {

   using value_type = typename std::common_type<
      Scalar1, MatrixA::Scalar, MatrixA::Scalar>::type;

   static value_type eval(Scalar1 alpha, const MatrixA& A, const MatrixB& B,
                          Op_flag opA, Op_flag opB)
      {
         if (opA == Op_flag::None && opB == Op_flag::None) {
            return (alpha * A * B).sum();
         } else if (opA == Op_flag::Transpose && opB == Op_flag::None) {
            return (alpha * A.transpose() * B).sum();
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::None) {
            return (alpha * A.adjoint() * B).sum();
         } else if (opA == Op_flag::None && opB == Op_flag::Transpose) {
            return (alpha * A * B.transpose()).sum();
         } else if (opA == Op_flag::Transpose && opB == Op_flag::Transpose) {
            return (alpha * A.transpose() * B.transpose()).sum();
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::Transpose) {
            return (alpha * A.adjoint() * B.transpose()).sum();
         } else if (opA == Op_flag::None && opB == Op_flag::Adjoint) {
            return (alpha * A * B.adjoint()).sum();
         } else if (opA == Op_flag::Transpose && opB == Op_flag::Adjoint) {
            return (alpha * A.transpose() * B.adjoint()).sum();
         } else if (opA == Op_flag::Adjoint && opB == Op_flag::Adjoint) {
            return (alpha * A.adjoint() * B.adjoint()).sum();
         }
      }
};

template <class MatrixC, class MatrixA, class MatrixB, class ResidualMatrix>
struct matrix_residual_impl<
   MatrixC, MatrixA, MatrixB, ResidualMatrix,
   typename std::enable_if<is_eigen_matrix<MatrixC>::value &&
                           is_eigen_matrix<MatrixA>::value &&
                           is_eigen_matrix<MatrixB>::value &&
                           is_eigen_matrix<ResidualMatrix>::value>::type> {

   static void eval(const MatrixC& C, const MatrixA& A, const MatrixB& B,
                    ResidualMatrix& res)
      {
         res = C - A * B;
      }
};

template <class MatrixC, class MatrixA, class MatrixB>
struct matrix_residual_fro_norm_impl<
   MatrixC, MatrixA, MatrixB,
   typename std::enable_if<is_eigen_matrix<MatrixC>::value &&
                           is_eigen_matrix<MatrixA>::value &&
                           is_eigen_matrix<MatrixB>::value>::type> {
   using value_type = typename std::common_type<MatrixC::Scalar,
                                                MatrixA::Scalar,
                                                MatrixB::Scalar>::type;

   static value_type eval(const MatrixC& C, const MatrixA& A, const MatrixB& B)
      {
         return (C - A * B).norm();
      }
};

template <class MatrixA, class MatrixB>
struct solve_qr_impl<
   MatrixA, MatrixB,
   typename std::enable_if<is_eigen_matrix<MatrixA>::value &&
                           is_eigen_matrix<MatrixB>::value>::type> {

   using return_type = int;

   static return_type eval(const MatrixA& A, MatrixB& B)
      {
         B = A.colPivHouseholderQr().solve(B);
      }
};

} // namespace detail

} // namespace backends

} // namespace sda

#endif
