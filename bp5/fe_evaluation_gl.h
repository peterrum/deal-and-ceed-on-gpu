
#ifndef fe_evaluation_gl_h
#define fe_evaluation_gl_h


#include <deal.II/base/config.h>

#ifdef DEAL_II_COMPILER_CUDA_AWARE

#  include <deal.II/base/tensor.h>
#  include <deal.II/base/utilities.h>

#  include <deal.II/lac/cuda_atomic.h>
#  include <deal.II/lac/cuda_vector.h>

#  include <deal.II/matrix_free/cuda_hanging_nodes_internal.h>
#  include <deal.II/matrix_free/cuda_matrix_free.h>
#  include <deal.II/matrix_free/cuda_matrix_free.templates.h>
#  include <deal.II/matrix_free/cuda_tensor_product_kernels.h>


DEAL_II_NAMESPACE_OPEN

namespace CUDAWrappers
{
  template <int dim,
            int fe_degree,
            int n_q_points_1d = fe_degree + 1,
            int n_components_ = 1,
            typename Number   = double>
  class FEEvaluationGL
  {
  public:
    using value_type    = Number;
    using gradient_type = Tensor<1, dim, Number>;
    using data_type     = typename MatrixFree<dim, Number>::Data;
    static constexpr unsigned int dimension    = dim;
    static constexpr unsigned int n_components = n_components_;
    static constexpr unsigned int n_q_points =
      Utilities::pow(n_q_points_1d, dim);
    static constexpr unsigned int tensor_dofs_per_cell =
      Utilities::pow(fe_degree + 1, dim);

    __device__
    FEEvaluationGL(const unsigned int       cell_id,
		   const data_type *        data,
		   SharedData<dim, Number> *shdata);

    __device__ void
    read_dof_values(const Number *src);

    __device__ void
    distribute_local_to_global(Number *dst) const;

    __device__ void
    evaluate(const bool evaluate_val, const bool evaluate_grad);

    __device__ void
    integrate(const bool integrate_val, const bool integrate_grad);

    __device__ value_type
               get_value(const unsigned int q_point) const;

    __device__ value_type
               get_dof_value(const unsigned int dof) const;

    __device__ void
    submit_value(const value_type &val_in, const unsigned int q_point);

    __device__ void
    submit_dof_value(const value_type &val_in, const unsigned int dof);

    __device__ gradient_type
               get_gradient(const unsigned int q_point) const;

    __device__ void
    submit_gradient(const gradient_type &grad_in, const unsigned int q_point);

    template <typename Functor>
    __device__ void
    apply_quad_point_operations(const Functor &func);

  private:
    types::global_dof_index *local_to_global;
    unsigned int             n_cells;
    unsigned int             padding_length;

    const unsigned int constraint_mask;

    const bool use_coloring;

    Number *inv_jac;
    Number *JxW;

    // Internal buffer
    Number *values;
    Number *gradients[dim];
  };



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  __device__
  FEEvaluationGL<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    FEEvaluationGL(const unsigned int       cell_id,
		   const data_type *        data,
		   SharedData<dim, Number> *shdata)
    : n_cells(data->n_cells)
    , padding_length(data->padding_length)
    , constraint_mask(data->constraint_mask[cell_id])
    , use_coloring(data->use_coloring)
    , values(shdata->values)
  {
    local_to_global = data->local_to_global + padding_length * cell_id;
    inv_jac         = data->inv_jacobian + padding_length * cell_id;
    JxW             = data->JxW + padding_length * cell_id;

    for (unsigned int i = 0; i < dim; ++i)
      gradients[i] = shdata->gradients[i];
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  __device__ void
  FEEvaluationGL<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    read_dof_values(const Number *src)
  {
    static_assert(n_components_ == 1, "This function only supports FE with one \
                  components");
    const unsigned int idx =
      (threadIdx.x % n_q_points_1d) +
      (dim > 1 ? threadIdx.y : 0) * n_q_points_1d +
      (dim > 2 ? threadIdx.z : 0) * n_q_points_1d * n_q_points_1d;

    const types::global_dof_index src_idx = local_to_global[idx];
    // Use the read-only data cache.
    values[idx] = __ldg(&src[src_idx]);

    __syncthreads();

    internal::resolve_hanging_nodes<dim, fe_degree, false>(constraint_mask,
                                                           values);
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  __device__ void
  FEEvaluationGL<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    distribute_local_to_global(Number *dst) const
  {
    static_assert(n_components_ == 1, "This function only supports FE with one \
                  components");
    internal::resolve_hanging_nodes<dim, fe_degree, true>(constraint_mask,
                                                          values);

    const unsigned int idx =
      (threadIdx.x % n_q_points_1d) +
      (dim > 1 ? threadIdx.y : 0) * n_q_points_1d +
      (dim > 2 ? threadIdx.z : 0) * n_q_points_1d * n_q_points_1d;
    const types::global_dof_index destination_idx = local_to_global[idx];

    if (use_coloring)
      dst[destination_idx] += values[idx];
    else
      LinearAlgebra::CUDAWrappers::atomicAdd_wrapper(&dst[destination_idx],
                                                     values[idx]);
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  __device__ void
  FEEvaluationGL<dim, fe_degree, n_q_points_1d, n_components_, Number>::evaluate(
    const bool evaluate_val,
    const bool evaluate_grad)
  {
    // First evaluate the gradients because it requires values that will be
    // changed if evaluate_val is true
    internal::EvaluatorTensorProduct<
      internal::EvaluatorVariant::evaluate_general,
      dim,
      fe_degree,
      n_q_points_1d,
      Number>
      evaluator_tensor_product;
    if (evaluate_grad == true)
      {
        evaluator_tensor_product.gradient_at_quad_pts(values, gradients);
        __syncthreads();
      }

    if (evaluate_val == true)
      {
        evaluator_tensor_product.value_at_quad_pts(values);
        __syncthreads();
      }
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  __device__ void
  FEEvaluationGL<dim, fe_degree, n_q_points_1d, n_components_, Number>::integrate(
    const bool integrate_val,
    const bool integrate_grad)
  {
    internal::EvaluatorTensorProduct<
      internal::EvaluatorVariant::evaluate_general,
      dim,
      fe_degree,
      n_q_points_1d,
      Number>
      evaluator_tensor_product;
    if (integrate_val == true)
      {
        evaluator_tensor_product.integrate_value(values);
        __syncthreads();
        if (integrate_grad == true)
          {
            evaluator_tensor_product.integrate_gradient<true>(values,
                                                              gradients);
            __syncthreads();
          }
      }
    else if (integrate_grad == true)
      {
        evaluator_tensor_product.integrate_gradient<false>(values, gradients);
        __syncthreads();
      }
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  __device__ typename FEEvaluationGL<dim,
                                   fe_degree,
                                   n_q_points_1d,
                                   n_components_,
                                   Number>::value_type
  FEEvaluationGL<dim, fe_degree, n_q_points_1d, n_components_, Number>::get_value(
    const unsigned int q_point) const
  {
    return values[q_point];
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
    __device__ typename FEEvaluationGL<dim,
                                   fe_degree,
                                   n_q_points_1d,
                                   n_components_,
                                   Number>::value_type
  FEEvaluationGL<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    get_dof_value(const unsigned int dof) const
  {
    return values[dof];
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  __device__ void
  FEEvaluationGL<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    submit_value(const value_type &val_in, const unsigned int q_point)
  {
    values[q_point] = val_in * JxW[q_point];
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  __device__ void
  FEEvaluationGL<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    submit_dof_value(const value_type &val_in, const unsigned int dof)
  {
    values[dof] = val_in;
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  __device__ typename FEEvaluationGL<dim,
                                   fe_degree,
                                   n_q_points_1d,
                                   n_components_,
                                   Number>::gradient_type
  FEEvaluationGL<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    get_gradient(const unsigned int q_point) const
  {
    static_assert(n_components_ == 1, "This function only supports FE with one \
                  components");
    // TODO optimize if the mesh is uniform
    const Number *inv_jacobian = &inv_jac[q_point];
    gradient_type grad;
    for (int d_1 = 0; d_1 < dim; ++d_1)
      {
        Number tmp = 0.;
        for (int d_2 = 0; d_2 < dim; ++d_2)
          tmp += inv_jacobian[padding_length * n_cells * (dim * d_2 + d_1)] *
                 gradients[d_2][q_point];
        grad[d_1] = tmp;
      }

    return grad;
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  __device__ void
  FEEvaluationGL<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    submit_gradient(const gradient_type &grad_in, const unsigned int q_point)
  {
    // TODO optimize if the mesh is uniform
    const Number *inv_jacobian = &inv_jac[q_point];
    for (int d_1 = 0; d_1 < dim; ++d_1)
      {
        Number tmp = 0.;
        for (int d_2 = 0; d_2 < dim; ++d_2)
          tmp += inv_jacobian[n_cells * padding_length * (dim * d_1 + d_2)] *
                 grad_in[d_2];
        gradients[d_1][q_point] = tmp * JxW[q_point];
      }
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components_,
            typename Number>
  template <typename Functor>
  __device__ void
  FEEvaluationGL<dim, fe_degree, n_q_points_1d, n_components_, Number>::
    apply_quad_point_operations(const Functor &func)
  {
    const unsigned int q_point =
      (dim == 1 ?
         threadIdx.x % n_q_points_1d :
         dim == 2 ?
         threadIdx.x % n_q_points_1d + n_q_points_1d * threadIdx.y :
         threadIdx.x % n_q_points_1d +
             n_q_points_1d * (threadIdx.y + n_q_points_1d * threadIdx.z));
    func(this, q_point);

    __syncthreads();
  }
} // namespace CUDAWrappers

DEAL_II_NAMESPACE_CLOSE

#endif

#endif
