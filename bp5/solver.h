#include <deal.II/base/exceptions.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/subscriptor.h>
#include <deal.II/base/tensor.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/solver.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/tridiagonal_matrix.h>

#include <cmath>

DEAL_II_NAMESPACE_OPEN

// forward declaration
#ifndef DOXYGEN
class PreconditionIdentity;
#endif

//#define IGNORE_FAILURE


/*!@addtogroup Solvers */
/*@{*/

/**
 * This class implements the preconditioned Conjugate Gradients (CG)
 * method that can be used to solve linear systems with a symmetric positive
 * definite matrix. This
 * class is used first in step-3 and step-4, but is used in many other
 * tutorial programs as well. Like all other solver classes, it can work on
 * any kind of vector and matrix as long as they satisfy certain requirements
 * (for the requirements on matrices and vectors in order to work with this
 * class, see the documentation of the Solver base class). The type of the
 * solution vector must be passed as template argument, and defaults to
 * dealii::Vector<double>.
 *
 * @note This version of CG is taken from D. Braess's book "Finite Elements".
 * It requires a symmetric preconditioner (i.e., for example, SOR is not a
 * possible choice).
 *
 *
 * <h3>Eigenvalue computation</h3>
 *
 * The cg-method performs an orthogonal projection of the original
 * preconditioned linear system to another system of smaller dimension.
 * Furthermore, the projected matrix @p T is tri-diagonal. Since the
 * projection is orthogonal, the eigenvalues of @p T approximate those of the
 * original preconditioned matrix @p PA. In fact, after @p n steps, where @p n
 * is the dimension of the original system, the eigenvalues of both matrices
 * are equal. But, even for small numbers of iteration steps, the condition
 * number of @p T is a good estimate for the one of @p PA.
 *
 * After @p m steps the matrix T_m can be written in terms of the coefficients
 * @p alpha and @p beta as the tri-diagonal matrix with diagonal elements
 * <tt>1/alpha_0</tt>, <tt>1/alpha_1 + beta_0/alpha_0</tt>, ...,
 * <tt>1/alpha_{m-1</tt>+beta_{m-2}/alpha_{m-2}} and off-diagonal elements
 * <tt>sqrt(beta_0)/alpha_0</tt>, ..., <tt>sqrt(beta_{m-2</tt>)/alpha_{m-2}}.
 * The eigenvalues of this matrix can be computed by postprocessing.
 *
 * @see Y. Saad: "Iterative methods for Sparse Linear Systems", section 6.7.3
 * for details.
 *
 * The coefficients, eigenvalues and condition number (computed as the ratio
 * of the largest over smallest eigenvalue) can be obtained by connecting a
 * function as a slot to the solver using one of the functions @p
 * connect_coefficients_slot, @p connect_eigenvalues_slot and @p
 * connect_condition_number_slot. These slots will then be called from the
 * solver with the estimates as argument.
 *
 * <h3>Observing the progress of linear solver iterations</h3>
 *
 * The solve() function of this class uses the mechanism described in the
 * Solver base class to determine convergence. This mechanism can also be used
 * to observe the progress of the iteration.
 *
 *
 * @author W. Bangerth, G. Kanschat, R. Becker and F.-T. Suttmeier
 */
template <typename VectorType = Vector<double>>
class SolverCG2 : public SolverBase<VectorType>
{
public:
  /**
   * Declare type for container size.
   */
  using size_type = types::global_dof_index;

  /**
   * Standardized data struct to pipe additional data to the solver.
   * Here, it doesn't store anything but just exists for consistency
   * with the other solver classes.
   */
  struct AdditionalData
  {};

  /**
   * Constructor.
   */
  SolverCG2(SolverControl &           cn,
            VectorMemory<VectorType> &mem,
            const AdditionalData &    data = AdditionalData());

  /**
   * Constructor. Use an object of type GrowingVectorMemory as a default to
   * allocate memory.
   */
  SolverCG2(SolverControl &cn, const AdditionalData &data = AdditionalData());

  /**
   * Virtual destructor.
   */
  virtual ~SolverCG2() override = default;

  /**
   * Solve the linear system $Ax=b$ for x.
   */
  template <typename MatrixType, typename PreconditionerType>
  void
  solve(const MatrixType &        A,
        VectorType &              x,
        const VectorType &        b,
        const PreconditionerType &preconditioner);

  /**
   * Connect a slot to retrieve the CG coefficients. The slot will be called
   * with alpha as the first argument and with beta as the second argument,
   * where alpha and beta follow the notation in Y. Saad: "Iterative methods
   * for Sparse Linear Systems", section 6.7. Called once per iteration
   */
  boost::signals2::connection
  connect_coefficients_slot(
    const std::function<void(typename VectorType::value_type,
                             typename VectorType::value_type)> &slot);

  /**
   * Connect a slot to retrieve the estimated condition number. Called on each
   * iteration if every_iteration=true, otherwise called once when iterations
   * are ended (i.e., either because convergence has been achieved, or because
   * divergence has been detected).
   */
  boost::signals2::connection
  connect_condition_number_slot(const std::function<void(double)> &slot,
                                const bool every_iteration = false);

  /**
   * Connect a slot to retrieve the estimated eigenvalues. Called on each
   * iteration if every_iteration=true, otherwise called once when iterations
   * are ended (i.e., either because convergence has been achieved, or because
   * divergence has been detected).
   */
  boost::signals2::connection
  connect_eigenvalues_slot(
    const std::function<void(const std::vector<double> &)> &slot,
    const bool every_iteration = false);

protected:
  /**
   * Interface for derived class. This function gets the current iteration
   * vector, the residual and the update vector in each step. It can be used
   * for graphical output of the convergence history.
   */
  virtual void
  print_vectors(const unsigned int step,
                const VectorType & x,
                const VectorType & r,
                const VectorType & d) const;

  /**
   * Estimates the eigenvalues from diagonal and offdiagonal. Uses these
   * estimate to compute the condition number. Calls the signals
   * eigenvalues_signal and cond_signal with these estimates as arguments.
   */
  static void
  compute_eigs_and_cond(
    const std::vector<typename VectorType::value_type> &diagonal,
    const std::vector<typename VectorType::value_type> &offdiagonal,
    const boost::signals2::signal<void(const std::vector<double> &)>
      &                                          eigenvalues_signal,
    const boost::signals2::signal<void(double)> &cond_signal);

  /**
   * Additional parameters.
   */
  AdditionalData additional_data;

  /**
   * Signal used to retrieve the CG coefficients. Called on each iteration.
   */
  boost::signals2::signal<void(typename VectorType::value_type,
                               typename VectorType::value_type)>
    coefficients_signal;

  /**
   * Signal used to retrieve the estimated condition number. Called once when
   * all iterations are ended.
   */
  boost::signals2::signal<void(double)> condition_number_signal;

  /**
   * Signal used to retrieve the estimated condition numbers. Called on each
   * iteration.
   */
  boost::signals2::signal<void(double)> all_condition_numbers_signal;

  /**
   * Signal used to retrieve the estimated eigenvalues. Called once when all
   * iterations are ended.
   */
  boost::signals2::signal<void(const std::vector<double> &)> eigenvalues_signal;

  /**
   * Signal used to retrieve the estimated eigenvalues. Called on each
   * iteration.
   */
  boost::signals2::signal<void(const std::vector<double> &)>
    all_eigenvalues_signal;
};

/*@}*/

/*------------------------- Implementation ----------------------------*/

#ifndef DOXYGEN

template <typename VectorType>
SolverCG2<VectorType>::SolverCG2(SolverControl &           cn,
                                 VectorMemory<VectorType> &mem,
                                 const AdditionalData &    data)
  : SolverBase<VectorType>(cn, mem)
  , additional_data(data)
{}



template <typename VectorType>
SolverCG2<VectorType>::SolverCG2(SolverControl &cn, const AdditionalData &data)
  : SolverBase<VectorType>(cn)
  , additional_data(data)
{}



template <typename VectorType>
void
SolverCG2<VectorType>::print_vectors(const unsigned int,
                                     const VectorType &,
                                     const VectorType &,
                                     const VectorType &) const
{}



template <typename VectorType>
inline void
SolverCG2<VectorType>::compute_eigs_and_cond(
  const std::vector<typename VectorType::value_type> &diagonal,
  const std::vector<typename VectorType::value_type> &offdiagonal,
  const boost::signals2::signal<void(const std::vector<double> &)>
    &                                          eigenvalues_signal,
  const boost::signals2::signal<void(double)> &cond_signal)
{
  // Avoid computing eigenvalues unless they are needed.
  if (!cond_signal.empty() || !eigenvalues_signal.empty())
    {
      TridiagonalMatrix<typename VectorType::value_type> T(diagonal.size(),
                                                           true);
      for (size_type i = 0; i < diagonal.size(); ++i)
        {
          T(i, i) = diagonal[i];
          if (i < diagonal.size() - 1)
            T(i, i + 1) = offdiagonal[i];
        }
      T.compute_eigenvalues();
      // Need two eigenvalues to estimate the condition number.
      if (diagonal.size() > 1)
        {
          auto condition_number = T.eigenvalue(T.n() - 1) / T.eigenvalue(0);
          // Condition number is real valued and nonnegative; simply take
          // the absolute value:
          cond_signal(std::abs(condition_number));
        }
      // Avoid copying the eigenvalues of T to a vector unless a signal is
      // connected.
      if (!eigenvalues_signal.empty())
        {
          std::vector<double> eigenvalues(T.n());
          for (unsigned int j = 0; j < T.n(); ++j)
            {
              // for a hermitian matrix, all eigenvalues are real-valued
              // and non-negative, simply return the absolute value:
              eigenvalues[j] = std::abs(T.eigenvalue(j));
            }
          eigenvalues_signal(eigenvalues);
        }
    }
}

namespace my_kernel
{
  using size_type = types::global_dof_index;

  template <typename Number>
  __global__ void
  update_a0(Number *        p,
            Number *        r,
            Number *        v,
            Number *        x,
            const Number *  diag,
            const Number    alpha,
            const Number    beta,
            const size_type N)
  {
    const size_type idx_base =
      threadIdx.x +
      blockIdx.x * (blockDim.x * ::dealii::CUDAWrappers::chunk_size);
    for (unsigned int i = 0; i < ::dealii::CUDAWrappers::chunk_size; ++i)
      {
        const size_type idx = idx_base + i * ::dealii::CUDAWrappers::block_size;
        if (idx < N)
          {
            p[idx] = -diag[idx] * r[idx];
            v[idx] = 0.0;
          }
      }
  }

  template <typename Number>
  __global__ void
  update_a(Number *        p,
           Number *        r,
           Number *        v,
           Number *        x,
           const Number *  diag,
           const Number    alpha,
           const Number    beta,
           const size_type N)
  {
    const size_type idx_base =
      threadIdx.x +
      blockIdx.x * (blockDim.x * ::dealii::CUDAWrappers::chunk_size);
    for (unsigned int i = 0; i < ::dealii::CUDAWrappers::chunk_size; ++i)
      {
        const size_type idx = idx_base + i * ::dealii::CUDAWrappers::block_size;
        if (idx < N)
          {
            x[idx] += alpha * p[idx];
            r[idx] += alpha * v[idx];
            p[idx] = beta * p[idx] - diag[idx] * r[idx];
            v[idx] = 0.0;
          }
      }
  }

  template <typename Number>
  __global__ void
  update_b(Number *        result,
           const Number *  p,
           const Number *  r,
           const Number *  v,
           const Number *  diag,
           const size_type N)
  {
    __shared__ Number result_buffer0[::dealii::CUDAWrappers::block_size];
    __shared__ Number result_buffer1[::dealii::CUDAWrappers::block_size];
    __shared__ Number result_buffer2[::dealii::CUDAWrappers::block_size];
    __shared__ Number result_buffer3[::dealii::CUDAWrappers::block_size];
    __shared__ Number result_buffer4[::dealii::CUDAWrappers::block_size];
    __shared__ Number result_buffer5[::dealii::CUDAWrappers::block_size];
    __shared__ Number result_buffer6[::dealii::CUDAWrappers::block_size];

    const size_type global_idx =
      threadIdx.x +
      blockIdx.x * (blockDim.x * ::dealii::CUDAWrappers::chunk_size);
    const size_type local_idx = threadIdx.x;


    if (global_idx < N)
      {
        // clang-format off
        result_buffer0[local_idx] = p[global_idx] * v[global_idx];
        result_buffer1[local_idx] = v[global_idx] * v[global_idx];
        result_buffer2[local_idx] = r[global_idx] * v[global_idx];
        result_buffer3[local_idx] = r[global_idx] * r[global_idx];
        result_buffer4[local_idx] = r[global_idx] * diag[global_idx] * v[global_idx];
        result_buffer5[local_idx] = v[global_idx] * diag[global_idx] * v[global_idx];
        result_buffer6[local_idx] = r[global_idx] * diag[global_idx] * r[global_idx];
        // clang-format on
      }
    else
      {
        result_buffer0[local_idx] = Number();
        result_buffer1[local_idx] = Number();
        result_buffer2[local_idx] = Number();
        result_buffer3[local_idx] = Number();
        result_buffer4[local_idx] = Number();
        result_buffer5[local_idx] = Number();
        result_buffer6[local_idx] = Number();
      }

    for (unsigned int i = 1; i < ::dealii::CUDAWrappers::chunk_size; ++i)
      {
        const size_type idx =
          global_idx + i * ::dealii::CUDAWrappers::block_size;
        if (idx < N)
          {
            result_buffer0[local_idx] += p[idx] * v[idx];
            result_buffer1[local_idx] += v[idx] * v[idx];
            result_buffer2[local_idx] += r[idx] * v[idx];
            result_buffer3[local_idx] += r[idx] * r[idx];
            result_buffer4[local_idx] += r[idx] * diag[idx] * v[idx];
            result_buffer5[local_idx] += v[idx] * diag[idx] * v[idx];
            result_buffer6[local_idx] += r[idx] * diag[idx] * r[idx];
          }
      }

    __syncthreads();

    for (size_type s = ::dealii::CUDAWrappers::block_size / 2; s > 32;
         s           = s >> 1)
      {
        if (local_idx < s)
          {
            result_buffer0[local_idx] += result_buffer0[local_idx + s];
            result_buffer1[local_idx] += result_buffer1[local_idx + s];
            result_buffer2[local_idx] += result_buffer2[local_idx + s];
            result_buffer3[local_idx] += result_buffer3[local_idx + s];
            result_buffer4[local_idx] += result_buffer4[local_idx + s];
            result_buffer5[local_idx] += result_buffer5[local_idx + s];
            result_buffer6[local_idx] += result_buffer6[local_idx + s];
          }
        __syncthreads();
      }

    if (local_idx < 32)
      if (::dealii::CUDAWrappers::block_size >= 64)
        {
          result_buffer0[local_idx] += result_buffer0[local_idx + 32];
          result_buffer1[local_idx] += result_buffer1[local_idx + 32];
          result_buffer2[local_idx] += result_buffer2[local_idx + 32];
          result_buffer3[local_idx] += result_buffer3[local_idx + 32];
          result_buffer4[local_idx] += result_buffer4[local_idx + 32];
          result_buffer5[local_idx] += result_buffer5[local_idx + 32];
          result_buffer6[local_idx] += result_buffer6[local_idx + 32];
        }
    if (local_idx < 16)
      if (::dealii::CUDAWrappers::block_size >= 32)
        {
          result_buffer0[local_idx] += result_buffer0[local_idx + 16];
          result_buffer1[local_idx] += result_buffer1[local_idx + 16];
          result_buffer2[local_idx] += result_buffer2[local_idx + 16];
          result_buffer3[local_idx] += result_buffer3[local_idx + 16];
          result_buffer4[local_idx] += result_buffer4[local_idx + 16];
          result_buffer5[local_idx] += result_buffer5[local_idx + 16];
          result_buffer6[local_idx] += result_buffer6[local_idx + 16];
        }
    if (local_idx < 8)
      if (::dealii::CUDAWrappers::block_size >= 16)
        {
          result_buffer0[local_idx] += result_buffer0[local_idx + 8];
          result_buffer1[local_idx] += result_buffer1[local_idx + 8];
          result_buffer2[local_idx] += result_buffer2[local_idx + 8];
          result_buffer3[local_idx] += result_buffer3[local_idx + 8];
          result_buffer4[local_idx] += result_buffer4[local_idx + 8];
          result_buffer5[local_idx] += result_buffer5[local_idx + 8];
          result_buffer6[local_idx] += result_buffer6[local_idx + 8];
        }
    if (local_idx < 4)
      if (::dealii::CUDAWrappers::block_size >= 8)
        {
          result_buffer0[local_idx] += result_buffer0[local_idx + 4];
          result_buffer1[local_idx] += result_buffer1[local_idx + 4];
          result_buffer2[local_idx] += result_buffer2[local_idx + 4];
          result_buffer3[local_idx] += result_buffer3[local_idx + 4];
          result_buffer4[local_idx] += result_buffer4[local_idx + 4];
          result_buffer5[local_idx] += result_buffer5[local_idx + 4];
          result_buffer6[local_idx] += result_buffer6[local_idx + 4];
        }
    if (local_idx < 2)
      if (::dealii::CUDAWrappers::block_size >= 4)
        {
          result_buffer0[local_idx] += result_buffer0[local_idx + 2];
          result_buffer1[local_idx] += result_buffer1[local_idx + 2];
          result_buffer2[local_idx] += result_buffer2[local_idx + 2];
          result_buffer3[local_idx] += result_buffer3[local_idx + 2];
          result_buffer4[local_idx] += result_buffer4[local_idx + 2];
          result_buffer5[local_idx] += result_buffer5[local_idx + 2];
          result_buffer6[local_idx] += result_buffer6[local_idx + 2];
        }
    if (local_idx < 1)
      if (::dealii::CUDAWrappers::block_size >= 2)
        {
          result_buffer0[local_idx] += result_buffer0[local_idx + 1];
          result_buffer1[local_idx] += result_buffer1[local_idx + 1];
          result_buffer2[local_idx] += result_buffer2[local_idx + 1];
          result_buffer3[local_idx] += result_buffer3[local_idx + 1];
          result_buffer4[local_idx] += result_buffer4[local_idx + 1];
          result_buffer5[local_idx] += result_buffer5[local_idx + 1];
          result_buffer6[local_idx] += result_buffer6[local_idx + 1];
        }

    if (local_idx == 0)
      {
        atomicAdd(result + 0, result_buffer0[0]);
        atomicAdd(result + 1, result_buffer1[0]);
        atomicAdd(result + 2, result_buffer2[0]);
        atomicAdd(result + 3, result_buffer3[0]);
        atomicAdd(result + 4, result_buffer4[0]);
        atomicAdd(result + 5, result_buffer5[0]);
        atomicAdd(result + 6, result_buffer6[0]);
      }
  }

} // namespace my_kernel



template <typename VectorType>
template <typename MatrixType, typename PreconditionerType>
void
SolverCG2<VectorType>::solve(const MatrixType &        A,
                             VectorType &              x,
                             const VectorType &        b,
                             const PreconditionerType &preconditioner)
{
  dealii::SolverControl::State conv = dealii::SolverControl::iterate;
  using number                      = typename VectorType::value_type;

  // Memory allocation
  typename dealii::VectorMemory<VectorType>::Pointer g_pointer(this->memory);
  typename dealii::VectorMemory<VectorType>::Pointer d_pointer(this->memory);
  typename dealii::VectorMemory<VectorType>::Pointer h_pointer(this->memory);

  // define some aliases for simpler access
  VectorType &g = *g_pointer;
  VectorType &d = *d_pointer;
  VectorType &h = *h_pointer;

  int    it       = 0;
  number res_norm = -std::numeric_limits<number>::max();

  // resize the vectors, but do not set the values since they'd be
  // overwritten soon anyway.
  g.reinit(x /*, true*/);
  d.reinit(x /*, true*/);
  h.reinit(x /*, true*/);

  // compute residual. if vector is zero, then short-circuit the full
  // computation
  if (!x.all_zero())
    {
      A.vmult(g, x);
      g.add(-1., b);
    }
  else
    g.equ(-1., b);
  res_norm = g.l2_norm();

  conv = this->iteration_status(0, res_norm, x);
  if (conv != dealii::SolverControl::iterate)
    return;

  number alpha = 0.0;
  number beta  = 0.0;

  number *    results_dev;
  cudaError_t error_code = cudaMalloc(&results_dev, 7 * sizeof(number));
  AssertCuda(error_code);

  while (conv == dealii::SolverControl::iterate)
    {
      it++;

      using size_type = types::global_dof_index;
      const int n_blocks =
        1 + x.local_size() / (::dealii::CUDAWrappers::chunk_size *
                              ::dealii::CUDAWrappers::block_size);

      // clear stash for dot-region
      error_code = cudaMemsetAsync(results_dev, 0, 7 * sizeof(number));
      AssertCuda(error_code);

      // 1) update region
      if (alpha == 0.0)
        my_kernel::update_a0<number>
          <<<n_blocks, ::dealii::CUDAWrappers::block_size>>>(
            d.get_values(),
            g.get_values(),
            h.get_values(),
            x.get_values(),
            preconditioner.get_vector().get_values(),
            alpha,
            beta,
            x.local_size());
      else
        my_kernel::update_a<number>
          <<<n_blocks, ::dealii::CUDAWrappers::block_size>>>(
            d.get_values(),
            g.get_values(),
            h.get_values(),
            x.get_values(),
            preconditioner.get_vector().get_values(),
            alpha,
            beta,
            x.local_size());

      // 2) matrix vector multiplication
      A.vmult(h, d);

      // 3) dot-production region
      my_kernel::update_b<number>
        <<<dim3(n_blocks, 1), dim3(::dealii::CUDAWrappers::block_size)>>>(
          results_dev,
          d.get_values(),
          g.get_values(),
          h.get_values(),
          preconditioner.get_vector().get_values(),
          x.local_size());

      // 4) compute scalars
      number results[7];
      cudaMemcpy(results,
                 results_dev,
                 7 * sizeof(number),
                 cudaMemcpyDeviceToHost);
      MPI_Allreduce(
        MPI_IN_PLACE, results, 7, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

      Assert(std::abs(results[0]) != 0., dealii::ExcDivideByZero());
      alpha = results[6] / results[0];

      res_norm = std::sqrt(results[3] + 2 * alpha * results[2] +
                           alpha * alpha * results[1]);
      conv     = this->iteration_status(it, res_norm, x);
      if (conv != dealii::SolverControl::iterate)
        {
          x.add(alpha, d);
          break;
        }

      beta = alpha * (results[4] + alpha * results[5]) / results[6];
    }

  cudaFree(results_dev);

  // in case of failure: throw exception
  if (conv != dealii::SolverControl::success)
    AssertThrow(false, dealii::SolverControl::NoConvergence(it, res_norm));
  // otherwise exit as normal
}



template <typename VectorType>
boost::signals2::connection
SolverCG2<VectorType>::connect_coefficients_slot(
  const std::function<void(typename VectorType::value_type,
                           typename VectorType::value_type)> &slot)
{
  return coefficients_signal.connect(slot);
}



template <typename VectorType>
boost::signals2::connection
SolverCG2<VectorType>::connect_condition_number_slot(
  const std::function<void(double)> &slot,
  const bool                         every_iteration)
{
  if (every_iteration)
    {
      return all_condition_numbers_signal.connect(slot);
    }
  else
    {
      return condition_number_signal.connect(slot);
    }
}



template <typename VectorType>
boost::signals2::connection
SolverCG2<VectorType>::connect_eigenvalues_slot(
  const std::function<void(const std::vector<double> &)> &slot,
  const bool                                              every_iteration)
{
  if (every_iteration)
    {
      return all_eigenvalues_signal.connect(slot);
    }
  else
    {
      return eigenvalues_signal.connect(slot);
    }
}



#endif // DOXYGEN

DEAL_II_NAMESPACE_CLOSE