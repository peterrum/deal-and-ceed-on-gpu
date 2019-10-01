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

template <typename VectorType = Vector<double>>
class SolverCG2 : public SolverBase<VectorType>
{
public:
  using size_type = types::global_dof_index;

  SolverCG2(SolverControl &cn);

  virtual ~SolverCG2() override = default;

  template <typename MatrixType, typename PreconditionerType>
  void
  solve(const MatrixType &        A,
        VectorType &              x,
        const VectorType &        b,
        const PreconditionerType &preconditioner);
};



template <typename VectorType>
SolverCG2<VectorType>::SolverCG2(SolverControl &cn)
  : SolverBase<VectorType>(cn)
{}



namespace internal
{
  namespace kernels
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
          const size_type idx =
            idx_base + i * ::dealii::CUDAWrappers::block_size;
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
          const size_type idx =
            idx_base + i * ::dealii::CUDAWrappers::block_size;
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

  } // namespace kernels
} // namespace internal



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
        internal::kernels::update_a0<number>
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
        internal::kernels::update_a<number>
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
      internal::kernels::update_b<number>
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


DEAL_II_NAMESPACE_CLOSE