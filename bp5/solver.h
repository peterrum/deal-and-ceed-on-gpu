#include <deal.II/base/tensor.h>
#include <deal.II/lac/diagonal_matrix.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/subscriptor.h>

#include <deal.II/lac/solver.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/tridiagonal_matrix.h>

#include <cmath>

DEAL_II_NAMESPACE_OPEN

// forward declaration
#ifndef DOXYGEN
class PreconditionIdentity;
#endif


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
using ::dealii::CUDAWrappers::block_size;
using ::dealii::CUDAWrappers::chunk_size;
using size_type = types::global_dof_index;

      template <typename Number>
      __global__ void
      update_a(Number *p, Number *r, Number *v, Number *x, Number * diag,  const Number alpha, const Number beta, const size_type N)
      {
          
          
        const size_type idx_base =
          threadIdx.x + blockIdx.x * (blockDim.x * chunk_size);
        for (unsigned int i = 0; i < chunk_size; ++i)
          {
            const size_type idx = idx_base + i * block_size;
            if (idx < N)
            {
              r[idx] = r[idx] - alpha * v[idx];
              x[idx] = x[idx] + alpha * p[idx];
              p[idx] = diag[idx] * r[idx] + beta * p[idx];
            }
          }
      }
      
      template <typename Number>
      __global__ void
      update_b(Number *result, const Number *p, const Number *r, const Number *v, const Number *diag, const size_type N)
      {
        __shared__ Number result_buffer0[block_size];
        __shared__ Number result_buffer1[block_size];
        __shared__ Number result_buffer2[block_size];
        __shared__ Number result_buffer3[block_size];
        __shared__ Number result_buffer4[block_size];
        __shared__ Number result_buffer5[block_size];
        __shared__ Number result_buffer6[block_size];

        const size_type global_idx =
          threadIdx.x + blockIdx.x * (blockDim.x * chunk_size);
        const size_type local_idx = threadIdx.x;

        if (global_idx < N)
        {
          result_buffer0[local_idx] = p[local_idx] * v[local_idx];
          result_buffer1[local_idx] = r[local_idx] * r[local_idx];
          result_buffer2[local_idx] = r[local_idx] * v[local_idx];
          result_buffer3[local_idx] = v[local_idx] * v[local_idx];
          result_buffer4[local_idx] = r[local_idx] * diag[local_idx] * r[local_idx];
          result_buffer5[local_idx] = r[local_idx] * diag[local_idx] * v[local_idx];
          result_buffer6[local_idx] = v[local_idx] * diag[local_idx] * v[local_idx];
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

        __syncthreads();
        
        for (size_type s = block_size / 2; s > 32; s = s >> 1)
          {
            if (local_idx < s)
            {
              result_buffer0[local_idx] = (result_buffer0[local_idx] + result_buffer0[local_idx + s]);
              result_buffer1[local_idx] = (result_buffer1[local_idx] + result_buffer1[local_idx + s]);
              result_buffer2[local_idx] = (result_buffer2[local_idx] + result_buffer2[local_idx + s]);
              result_buffer3[local_idx] = (result_buffer3[local_idx] + result_buffer3[local_idx + s]);
              result_buffer4[local_idx] = (result_buffer4[local_idx] + result_buffer4[local_idx + s]);
              result_buffer5[local_idx] = (result_buffer5[local_idx] + result_buffer5[local_idx + s]);
              result_buffer6[local_idx] = (result_buffer6[local_idx] + result_buffer6[local_idx + s]);
            }
            __syncthreads();
          }

        if (local_idx < 32)
        {
          if (block_size >= 64)
          {
            result_buffer0[local_idx] = (result_buffer0[local_idx] + result_buffer0[local_idx + 32]);
            result_buffer1[local_idx] = (result_buffer1[local_idx] + result_buffer1[local_idx + 32]);
            result_buffer2[local_idx] = (result_buffer2[local_idx] + result_buffer2[local_idx + 32]);
            result_buffer3[local_idx] = (result_buffer3[local_idx] + result_buffer3[local_idx + 32]);
            result_buffer4[local_idx] = (result_buffer4[local_idx] + result_buffer4[local_idx + 32]);
            result_buffer5[local_idx] = (result_buffer5[local_idx] + result_buffer5[local_idx + 32]);
            result_buffer6[local_idx] = (result_buffer6[local_idx] + result_buffer6[local_idx + 32]);
          }
          if (block_size >= 32)
          {
            result_buffer0[local_idx] = (result_buffer0[local_idx] + result_buffer0[local_idx + 16]);
            result_buffer1[local_idx] = (result_buffer1[local_idx] + result_buffer1[local_idx + 16]);
            result_buffer2[local_idx] = (result_buffer2[local_idx] + result_buffer2[local_idx + 16]);
            result_buffer3[local_idx] = (result_buffer3[local_idx] + result_buffer3[local_idx + 16]);
            result_buffer4[local_idx] = (result_buffer4[local_idx] + result_buffer4[local_idx + 16]);
            result_buffer5[local_idx] = (result_buffer5[local_idx] + result_buffer5[local_idx + 16]);
            result_buffer6[local_idx] = (result_buffer6[local_idx] + result_buffer6[local_idx + 16]);
          }
          if (block_size >= 16)
          {
            result_buffer0[local_idx] = (result_buffer0[local_idx] + result_buffer0[local_idx + 8]);
            result_buffer1[local_idx] = (result_buffer1[local_idx] + result_buffer1[local_idx + 8]);
            result_buffer2[local_idx] = (result_buffer2[local_idx] + result_buffer2[local_idx + 8]);
            result_buffer3[local_idx] = (result_buffer3[local_idx] + result_buffer3[local_idx + 8]);
            result_buffer4[local_idx] = (result_buffer4[local_idx] + result_buffer4[local_idx + 8]);
            result_buffer5[local_idx] = (result_buffer5[local_idx] + result_buffer5[local_idx + 8]);
            result_buffer6[local_idx] = (result_buffer6[local_idx] + result_buffer6[local_idx + 8]);
          }
          if (block_size >= 8)
          {
            result_buffer0[local_idx] = (result_buffer0[local_idx] + result_buffer0[local_idx + 4]);
            result_buffer1[local_idx] = (result_buffer1[local_idx] + result_buffer1[local_idx + 4]);
            result_buffer2[local_idx] = (result_buffer2[local_idx] + result_buffer2[local_idx + 4]);
            result_buffer3[local_idx] = (result_buffer3[local_idx] + result_buffer3[local_idx + 4]);
            result_buffer4[local_idx] = (result_buffer4[local_idx] + result_buffer4[local_idx + 4]);
            result_buffer5[local_idx] = (result_buffer5[local_idx] + result_buffer5[local_idx + 4]);
            result_buffer6[local_idx] = (result_buffer6[local_idx] + result_buffer6[local_idx + 4]);
          }
          if (block_size >= 4)
          {
            result_buffer0[local_idx] = (result_buffer0[local_idx] + result_buffer0[local_idx + 2]);
            result_buffer1[local_idx] = (result_buffer1[local_idx] + result_buffer1[local_idx + 2]);
            result_buffer2[local_idx] = (result_buffer2[local_idx] + result_buffer2[local_idx + 2]);
            result_buffer3[local_idx] = (result_buffer3[local_idx] + result_buffer3[local_idx + 2]);
            result_buffer4[local_idx] = (result_buffer4[local_idx] + result_buffer4[local_idx + 2]);
            result_buffer5[local_idx] = (result_buffer5[local_idx] + result_buffer5[local_idx + 2]);
            result_buffer6[local_idx] = (result_buffer6[local_idx] + result_buffer6[local_idx + 2]);
          }
          if (block_size >= 2)
          {
            result_buffer0[local_idx] = (result_buffer0[local_idx] + result_buffer0[local_idx + 1]);
            result_buffer1[local_idx] = (result_buffer1[local_idx] + result_buffer1[local_idx + 1]);
            result_buffer2[local_idx] = (result_buffer2[local_idx] + result_buffer2[local_idx + 1]);
            result_buffer3[local_idx] = (result_buffer3[local_idx] + result_buffer3[local_idx + 1]);
            result_buffer4[local_idx] = (result_buffer4[local_idx] + result_buffer4[local_idx + 1]);
            result_buffer5[local_idx] = (result_buffer5[local_idx] + result_buffer5[local_idx + 1]);
            result_buffer6[local_idx] = (result_buffer6[local_idx] + result_buffer6[local_idx + 1]);
          }
        }

        if (local_idx == 0)
        {
          LinearAlgebra::CUDAWrappers::atomicAdd_wrapper(result+0, result_buffer0[0]);
          LinearAlgebra::CUDAWrappers::atomicAdd_wrapper(result+1, result_buffer1[0]);
          LinearAlgebra::CUDAWrappers::atomicAdd_wrapper(result+2, result_buffer2[0]);
          LinearAlgebra::CUDAWrappers::atomicAdd_wrapper(result+3, result_buffer3[0]);
          LinearAlgebra::CUDAWrappers::atomicAdd_wrapper(result+4, result_buffer4[0]);
          LinearAlgebra::CUDAWrappers::atomicAdd_wrapper(result+5, result_buffer5[0]);
          LinearAlgebra::CUDAWrappers::atomicAdd_wrapper(result+6, result_buffer6[0]);
        }
      }      
      
}
      
      


template <typename VectorType>
template <typename MatrixType, typename PreconditionerType>
void
SolverCG2<VectorType>::solve(const MatrixType &        A,
                             VectorType &              x,
                             const VectorType &        b,
                             const PreconditionerType &preconditioner)
{
#ifdef BLUB
  using number = typename VectorType::value_type;

  SolverControl::State conv = SolverControl::iterate;

  LogStream::Prefix prefix("cg");
  

  // Memory allocation
  typename VectorMemory<VectorType>::Pointer g_pointer(this->memory);
  typename VectorMemory<VectorType>::Pointer d_pointer(this->memory);
  typename VectorMemory<VectorType>::Pointer h_pointer(this->memory);

  // define some aliases for simpler access
  VectorType &g = *g_pointer;
  VectorType &d = *d_pointer;
  VectorType &h = *h_pointer;

  // Should we build the matrix for eigenvalue computations?
  const bool do_eigenvalues =
    !condition_number_signal.empty() || !all_condition_numbers_signal.empty() ||
    !eigenvalues_signal.empty() || !all_eigenvalues_signal.empty();

  // vectors used for eigenvalue
  // computations
  std::vector<typename VectorType::value_type> diagonal;
  std::vector<typename VectorType::value_type> offdiagonal;

  int    it  = 0;
  double res = -std::numeric_limits<double>::max();

  typename VectorType::value_type eigen_beta_alpha = 0;

  // resize the vectors, but do not set
  // the values since they'd be overwritten
  // soon anyway.
  g.reinit(x, true);
  d.reinit(x, true);
  h.reinit(x, true);

  number gh, beta;

  // compute residual. if vector is
  // zero, then short-circuit the
  // full computation
  if (!x.all_zero())
    {
      A.vmult(g, x);
      g.add(-1., b);
    }
  else
    g.equ(-1., b);
  res = g.l2_norm();

  conv = this->iteration_status(0, res, x);
  if (conv != SolverControl::iterate)
    return;

  if (std::is_same<PreconditionerType, PreconditionIdentity>::value == false)
    {
      preconditioner.vmult(h, g);

      d.equ(-1., h);

      gh = g * h;
    }
  else
    {
      d.equ(-1., g);
      gh = res * res;
    }

  while (conv == SolverControl::iterate)
    {
      it++;
      A.vmult(h, d);

      number alpha = d * h;
      Assert(std::abs(alpha) != 0., ExcDivideByZero());
      alpha = gh / alpha;

      x.add(alpha, d);
      res = std::sqrt(std::abs(g.add_and_dot(alpha, h, g)));

      print_vectors(it, x, g, d);

      conv = this->iteration_status(it, res, x);
      if (conv != SolverControl::iterate)
        break;

      if (std::is_same<PreconditionerType, PreconditionIdentity>::value ==
          false)
        {
          preconditioner.vmult(h, g);

          beta = gh;
          Assert(std::abs(beta) != 0., ExcDivideByZero());
          gh   = g * h;
          beta = gh / beta;
          d.sadd(beta, -1., h);
        }
      else
        {
          beta = gh;
          gh   = res * res;
          beta = gh / beta;
          d.sadd(beta, -1., g);
        }

      this->coefficients_signal(alpha, beta);
      // set up the vectors
      // containing the diagonal
      // and the off diagonal of
      // the projected matrix.
      if (do_eigenvalues)
        {
          diagonal.push_back(number(1.) / alpha + eigen_beta_alpha);
          eigen_beta_alpha = beta / alpha;
          offdiagonal.push_back(std::sqrt(beta) / alpha);
        }
      compute_eigs_and_cond(diagonal,
                            offdiagonal,
                            all_eigenvalues_signal,
                            all_condition_numbers_signal);
    }

  compute_eigs_and_cond(diagonal,
                        offdiagonal,
                        eigenvalues_signal,
                        condition_number_signal);

  // in case of failure: throw exception
  if (conv != SolverControl::success)
    AssertThrow(false, SolverControl::NoConvergence(it, res));
  // otherwise exit as normal
  
#else
  
    dealii::SolverControl::State conv = dealii::SolverControl::iterate;
    using number = typename VectorType::value_type;

    // Memory allocation
    typename dealii::VectorMemory<VectorType>::Pointer g_pointer(this->memory);
    typename dealii::VectorMemory<VectorType>::Pointer d_pointer(this->memory);
    typename dealii::VectorMemory<VectorType>::Pointer h_pointer(this->memory);

    // define some aliases for simpler access
    VectorType &r = *g_pointer;
    VectorType &p = *d_pointer;
    VectorType &v = *h_pointer;

    int    it  = 0;
    double res_norm = -std::numeric_limits<double>::max();

    // resize the vectors, but do not set the values since they'd be
    // overwritten soon anyway.
    r.reinit(x, true);
    p.reinit(x, true);
    v.reinit(x, true);

    // compute residual. if vector is zero, then short-circuit the full
    // computation
    if (!x.all_zero())
      {
        A.vmult(r, x);
        r.add(-1., b);
      }
    else
      r.equ(-1., b);
    res_norm = r.l2_norm();

    conv = this->iteration_status(0, res_norm, x);
    if (conv != dealii::SolverControl::iterate)
      return;

  if (std::is_same<PreconditionerType, PreconditionIdentity>::value == false)
    {
      preconditioner.vmult(v, r);
      p.equ(-1., v);
    }
  else
    {
      p.equ(-1., r);
    }

    number alpha = 0.0; //(r * p) / (p * v);
    number beta  = 0.0;

    while (conv == dealii::SolverControl::iterate)
      {
        it++;
        
        using ::dealii::CUDAWrappers::block_size;
        using ::dealii::CUDAWrappers::chunk_size;
        using size_type = types::global_dof_index;
        
        const int n_blocks = 1 + x.size() / (chunk_size * block_size);
        my_kernel::update_a<double>
          <<<n_blocks, block_size>>>(p.get_values (), r.get_values (), 
                v.get_values (), x.get_values (), preconditioner.get_vector().get_values (), alpha, beta, x.size() );
        
        A.vmult(v, p);
        
        double results[7];
        
        my_kernel::update_b<double> <<<dim3(n_blocks, 1), dim3(block_size)>>>
            (results, p.get_values (), r.get_values (), v.get_values (), preconditioner.get_vector().get_values (), x.size());
        
        Assert(std::abs(results[0]) != 0., dealii::ExcDivideByZero());
        alpha = results[4] / results[0];

        res_norm = std::sqrt(results[1] + 2*alpha*results[2] + alpha*alpha*results[3]);
        conv = this->iteration_status(it, res_norm, x);
        if (conv != dealii::SolverControl::iterate)
          {
            x.add(alpha, p);
            break;
          }

        beta = (results[4] - 2 * alpha * results[5] + alpha * alpha * results[6])/results[4];
      }

    // in case of failure: throw exception
    if (conv != dealii::SolverControl::success)
      AssertThrow(false, dealii::SolverControl::NoConvergence(it, res_norm));
    // otherwise exit as normal
  
#endif
  
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