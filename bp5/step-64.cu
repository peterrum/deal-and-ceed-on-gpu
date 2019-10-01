/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2019 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/base/cuda.h>

#include <deal.II/matrix_free/cuda_fe_evaluation.h>
#include "fe_evaluation_gl.h"
#include <deal.II/matrix_free/cuda_matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <fstream>

#include "solver.h"


namespace Step64
{
  using namespace dealii;


  template <int dim, int fe_degree>
  class VaryingCoefficientFunctor
  {
  public:
    VaryingCoefficientFunctor(double *coefficient)
      : coef(coefficient)
    {}

    __device__ void operator()(
      const unsigned int                                          cell,
      const typename CUDAWrappers::MatrixFree<dim, double>::Data *gpu_data);

    static const unsigned int n_dofs_1d = fe_degree + 1;
    static const unsigned int n_local_dofs =
      dealii::Utilities::pow(n_dofs_1d, dim);
    static const unsigned int n_q_points =
      dealii::Utilities::pow(n_dofs_1d, dim);

  private:
    double *coef;
  };



  template <int dim, int fe_degree>
  __device__ void VaryingCoefficientFunctor<dim, fe_degree>::operator()(
    const unsigned int                                          cell,
    const typename CUDAWrappers::MatrixFree<dim, double>::Data *gpu_data)
  {
    const unsigned int pos = CUDAWrappers::local_q_point_id<dim, double>(
      cell, gpu_data, n_dofs_1d, n_q_points);
    const Point<dim> q_point =
      CUDAWrappers::get_quadrature_point<dim, double>(cell,
                                                      gpu_data,
                                                      n_dofs_1d);

    double p_square = 0.;
    for (unsigned int i = 0; i < dim; ++i)
      {
        const double coord = q_point[i];
        p_square += coord * coord;
      }
    coef[pos] = 10. / (0.05 + 2. * p_square);
  }


  template <int dim, int fe_degree>
  class HelmholtzOperatorQuad
  {
  public:
    __device__ HelmholtzOperatorQuad(double coef)
      : coef(coef)
    {}

    __device__ void
    operator()(CUDAWrappers::FEEvaluationGL<dim, fe_degree> *fe_eval,
               const unsigned int                          q) const;

  private:
    double coef;
  };


  template <int dim, int fe_degree>
  __device__ void HelmholtzOperatorQuad<dim, fe_degree>::
  operator()(CUDAWrappers::FEEvaluationGL<dim, fe_degree> *fe_eval,
             const unsigned int                          q) const
  {
    //fe_eval->submit_value(coef * fe_eval->get_value(q), q);
    fe_eval->submit_gradient(fe_eval->get_gradient(q), q);
  }


  template <int dim, int fe_degree>
  class LocalHelmholtzOperator
  {
  public:
    LocalHelmholtzOperator(double *coefficient)
      : coef(coefficient)
    {}

    __device__ void operator()(
      const unsigned int                                          cell,
      const typename CUDAWrappers::MatrixFree<dim, double>::Data *gpu_data,
      CUDAWrappers::SharedData<dim, double> *                     shared_data,
      const double *                                              src,
      double *                                                    dst) const;

    static const unsigned int n_dofs_1d    = fe_degree + 1;
    static const unsigned int n_local_dofs = Utilities::pow(fe_degree + 1, dim);
    static const unsigned int n_q_points   = Utilities::pow(fe_degree + 1, dim);

  private:
    double *coef;
  };


  template <int dim, int fe_degree>
  __device__ void LocalHelmholtzOperator<dim, fe_degree>::operator()(
    const unsigned int                                          cell,
    const typename CUDAWrappers::MatrixFree<dim, double>::Data *gpu_data,
    CUDAWrappers::SharedData<dim, double> *                     shared_data,
    const double *                                              src,
    double *                                                    dst) const
  {
    const unsigned int pos = CUDAWrappers::local_q_point_id<dim, double>(
      cell, gpu_data, n_dofs_1d, n_q_points);

    CUDAWrappers::FEEvaluationGL<dim, fe_degree, fe_degree + 1, 1, double>
      fe_eval(cell, gpu_data, shared_data);
    fe_eval.read_dof_values(src);
    fe_eval.evaluate(false, true);
    fe_eval.apply_quad_point_operations(
      HelmholtzOperatorQuad<dim, fe_degree>(coef[pos]));
    fe_eval.integrate(false, true);
    fe_eval.distribute_local_to_global(dst);
  }


  template <int dim, int fe_degree>
  class HelmholtzOperator
  {
  public:
    HelmholtzOperator(const DoFHandler<dim> &          dof_handler,
                      const AffineConstraints<double> &constraints);

    void
    vmult(LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> &dst,
          const LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA>
            &src) const;

    void
    initialize_dof_vector(LinearAlgebra::distributed::Vector<double,
    MemorySpace::CUDA> &vec) const
    {
      mf_data.initialize_dof_vector(vec);
    }

  private:
    CUDAWrappers::MatrixFree<dim, double>       mf_data;
    LinearAlgebra::CUDAWrappers::Vector<double> coef;
  };



  template <int dim, int fe_degree>
  HelmholtzOperator<dim, fe_degree>::HelmholtzOperator(
    const DoFHandler<dim> &          dof_handler,
    const AffineConstraints<double> &constraints)
  {
    MappingQGeneric<dim> mapping(fe_degree);
    typename CUDAWrappers::MatrixFree<dim, double>::AdditionalData
      additional_data;
    additional_data.mapping_update_flags = update_values | update_gradients |
                                           update_JxW_values |
                                           update_quadrature_points;
    const QGaussLobatto<1> quad(fe_degree + 1);
    mf_data.reinit(mapping, dof_handler, constraints, quad, additional_data);


    const unsigned int n_owned_cells =
      dynamic_cast<const parallel::Triangulation<dim> *>(
        &dof_handler.get_triangulation())
        ->n_locally_owned_active_cells();
    coef.reinit(Utilities::pow(fe_degree + 1, dim) * n_owned_cells);

    const VaryingCoefficientFunctor<dim, fe_degree> functor(coef.get_values());
    mf_data.evaluate_coefficients(functor);
  }


  template <int dim, int fe_degree>
  void HelmholtzOperator<dim, fe_degree>::vmult(
    LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> &      dst,
    const LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> &src)
    const
  {
    dst = 0.;
    LocalHelmholtzOperator<dim, fe_degree> helmholtz_operator(
      coef.get_values());
    mf_data.cell_loop(helmholtz_operator, src, dst);
    mf_data.copy_constrained_values(src, dst);
  }



  // Variant with merged coefficient tensor
  template <int dim, int fe_degree>
  class JacobianFunctor
  {
  public:
    JacobianFunctor(double *coefficient,
                    const unsigned int n_cells)
      : coef(coefficient)
      , n_cells(n_cells)
    {}

    __device__ void operator()
    (const unsigned int                                          cell,
     const typename CUDAWrappers::MatrixFree<dim, double>::Data *gpu_data);

    static const unsigned int n_dofs_1d = fe_degree + 1;
    static const unsigned int n_q_points = Utilities::pow(n_dofs_1d, dim);
  private:
    double *coef;
    const unsigned int n_cells;
  };


  template <int dim, int fe_degree>
  __device__ void JacobianFunctor<dim, fe_degree>::operator()
    (const unsigned int                                          cell,
     const typename CUDAWrappers::MatrixFree<dim, double>::Data *gpu_data)
  {
    const unsigned int q = CUDAWrappers::q_point_id_in_cell<dim>(fe_degree+1);
    Tensor<2,dim> inv_jac;
    for (unsigned int d=0; d<dim; ++d)
      for (unsigned int e=0; e<dim; ++e)
        inv_jac[d][e] = gpu_data->inv_jacobian[cell*gpu_data->padding_length + q
                                               + gpu_data->n_cells * gpu_data->padding_length * (d*dim+e)];
    Tensor<2,dim> my_coef;
    for (unsigned int d=0; d<dim; ++d)
      for (unsigned int e=d; e<dim; ++e)
        {
          double sum = inv_jac[d][0] * inv_jac[e][0];
          for (unsigned int f=1; f<dim; ++f)
            sum += inv_jac[d][f] * inv_jac[e][f];
          my_coef[d][e] = sum;
        }
    for (unsigned int d=0; d<dim; ++d)
      coef[q + cell*n_q_points + d*n_cells*n_q_points] =
        gpu_data->JxW[gpu_data->padding_length*cell+q] * my_coef[d][d];
    for (unsigned int c=dim, d=0; d<dim; ++d)
      for (unsigned int e=d+1; e<dim; ++e, ++c)
        coef[q + cell*n_q_points + c*n_cells*n_q_points] =
          gpu_data->JxW[gpu_data->padding_length*cell+q] * my_coef[d][e];
  }



  template <int dim, int fe_degree>
  class LocalHelmholtzOperatorMerged
  {
  public:
    LocalHelmholtzOperatorMerged(double *coefficient,
                                 const   unsigned int n_cells)
      : coef(coefficient)
      , n_cells(n_cells)
    {}

    __device__ void operator()(
      const unsigned int                                          cell,
      const typename CUDAWrappers::MatrixFree<dim, double>::Data *gpu_data,
      CUDAWrappers::SharedData<dim, double> *                     shared_data,
      const double *                                              src,
      double *                                                    dst) const;

    static const unsigned int n_dofs_1d    = fe_degree + 1;
    static const unsigned int n_local_dofs = Utilities::pow(fe_degree + 1, dim);
    static const unsigned int n_q_points   = Utilities::pow(fe_degree + 1, dim);

  private:
    const unsigned int n_cells;
    double *coef;
  };



  template <int dim, int fe_degree>
  __device__ void LocalHelmholtzOperatorMerged<dim, fe_degree>::operator()(
    const unsigned int                                          cell,
    const typename CUDAWrappers::MatrixFree<dim, double>::Data *gpu_data,
    CUDAWrappers::SharedData<dim, double> *                     shared_data,
    const double *                                              src,
    double *                                                    dst) const
  {
    const unsigned int pos = CUDAWrappers::local_q_point_id<dim, double>(
      cell, gpu_data, n_dofs_1d, n_q_points);
    const unsigned int offset = n_q_points * n_cells;

    CUDAWrappers::FEEvaluationGL<dim, fe_degree, fe_degree + 1, 1, double>
      fe_eval(cell, gpu_data, shared_data);
    fe_eval.read_dof_values(src);
    fe_eval.evaluate(false, true);
    const unsigned int q = CUDAWrappers::internal::compute_index<dim, fe_degree+1>();
    if (dim==3)
      {
        const double grad0 = shared_data->gradients[0][q];
        const double grad1 = shared_data->gradients[1][q];
        const double grad2 = shared_data->gradients[2][q];
        shared_data->gradients[0][q] = grad0 * coef[q] + grad1 * coef[q+3*offset] + grad2 * coef[q+4*offset];
        shared_data->gradients[1][q] = grad0 * coef[q+3*offset] + grad1 * coef[q+1*offset] + grad2 * coef[q+5*offset];
        shared_data->gradients[2][q] = grad0 * coef[q+4*offset] + grad1 * coef[q+5*offset] + grad2 * coef[q+2*offset];
      }
    else
      {
        const double grad0 = shared_data->gradients[0][q];
        const double grad1 = shared_data->gradients[1][q];
        shared_data->gradients[0][q] = grad0 * coef[q] + grad1 * coef[q+2*offset];
        shared_data->gradients[1][q] = grad0 * coef[q+2*offset] + grad1 * coef[q+1*offset];
      }
    __syncthreads();
    fe_eval.integrate(false, true);
    fe_eval.distribute_local_to_global(dst);
  }



  template <int dim, int fe_degree>
  class HelmholtzOperatorMerged
  {
  public:
    HelmholtzOperatorMerged(const DoFHandler<dim> &          dof_handler,
                            const AffineConstraints<double> &constraints);

    void
    vmult(LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> &dst,
          const LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA>
            &src) const;

    void
    initialize_dof_vector(LinearAlgebra::distributed::Vector<double,
    MemorySpace::CUDA> &vec) const
    {
      mf_data.initialize_dof_vector(vec);
    }

  private:
    CUDAWrappers::MatrixFree<dim, double>       mf_data;
    LinearAlgebra::CUDAWrappers::Vector<double> coef;
    unsigned int n_owned_cells;
  };



  template <int dim, int fe_degree>
  HelmholtzOperatorMerged<dim, fe_degree>::HelmholtzOperatorMerged(
    const DoFHandler<dim> &          dof_handler,
    const AffineConstraints<double> &constraints)
  {
    MappingQGeneric<dim> mapping(fe_degree);
    typename CUDAWrappers::MatrixFree<dim, double>::AdditionalData
      additional_data;
    additional_data.mapping_update_flags = update_values | update_gradients |
                                           update_JxW_values |
                                           update_quadrature_points;
    const QGaussLobatto<1> quad(fe_degree + 1);
    mf_data.reinit(mapping, dof_handler, constraints, quad, additional_data);

    n_owned_cells =
      dynamic_cast<const parallel::Triangulation<dim> *>(
        &dof_handler.get_triangulation())
        ->n_locally_owned_active_cells();
    coef.reinit(Utilities::pow(fe_degree + 1, dim) * n_owned_cells * dim * (dim+1) / 2);

    const JacobianFunctor<dim, fe_degree> functor(coef.get_values(), n_owned_cells);
    mf_data.evaluate_coefficients(functor);
  }



  template <int dim, int fe_degree>
  void HelmholtzOperatorMerged<dim, fe_degree>::vmult(
    LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> &      dst,
    const LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> &src)
    const
  {
    dst = 0.;
    LocalHelmholtzOperatorMerged<dim, fe_degree>
      helmholtz_operator(coef.get_values(), n_owned_cells);
    mf_data.cell_loop(helmholtz_operator, src, dst);
    mf_data.copy_constrained_values(src, dst);
  }



  template <int dim, int fe_degree>
  class HelmholtzProblem
  {
  public:
    HelmholtzProblem();

    void run();

  private:
    void setup_system();

    void assemble_rhs();

    void solve();

    void output_results(const unsigned int cycle) const;

    MPI_Comm mpi_communicator;

    parallel::distributed::Triangulation<dim> triangulation;

    FE_Q<dim>       fe;
    DoFHandler<dim> dof_handler;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    AffineConstraints<double>                          constraints;
    std::unique_ptr<HelmholtzOperatorMerged<dim, fe_degree>> system_matrix_dev;

    LinearAlgebra::distributed::Vector<double, MemorySpace::Host>
                                                                  ghost_solution_host;
    LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> solution_dev;
    LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA>
      system_rhs_dev;

    ConditionalOStream pcout;
  };


  template <int dim, int fe_degree>
  HelmholtzProblem<dim, fe_degree>::HelmholtzProblem()
    : mpi_communicator(MPI_COMM_WORLD)
    , triangulation(mpi_communicator)
    , fe(fe_degree)
    , dof_handler(triangulation)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
  {}



  template <int dim, int fe_degree>
  void HelmholtzProblem<dim, fe_degree>::setup_system()
  {
    dof_handler.distribute_dofs(fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    system_rhs_dev.reinit(locally_owned_dofs, mpi_communicator);

    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(),
                                             constraints);
    constraints.close();

    system_matrix_dev.reset(
      new HelmholtzOperatorMerged<dim, fe_degree>(dof_handler, constraints));

    ghost_solution_host.reinit(locally_owned_dofs,
                               locally_relevant_dofs,
                               mpi_communicator);
    system_matrix_dev->initialize_dof_vector(solution_dev);
    system_rhs_dev.reinit(solution_dev);
  }



  template <int dim, int fe_degree>
  void HelmholtzProblem<dim, fe_degree>::assemble_rhs()
  {
    LinearAlgebra::distributed::Vector<double, MemorySpace::Host>
                      system_rhs_host(locally_owned_dofs,
                      locally_relevant_dofs,
                      mpi_communicator);
    const QGauss<dim> quadrature_formula(fe_degree + 1);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_quadrature_points |
                              update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();

    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell_rhs = 0;

          fe_values.reinit(cell);

          for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
            {
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                cell_rhs(i) += (fe_values.shape_value(i, q_index) * 1.0 *
                                fe_values.JxW(q_index));
            }

          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(cell_rhs,
                                                 local_dof_indices,
                                                 system_rhs_host);
        }
    system_rhs_host.compress(VectorOperation::add);

    LinearAlgebra::ReadWriteVector<double> rw_vector(locally_owned_dofs);
    rw_vector.import(system_rhs_host, VectorOperation::insert);
    system_rhs_dev.import(rw_vector, VectorOperation::insert);
  }



  template <int dim, int fe_degree>
  void HelmholtzProblem<dim, fe_degree>::solve()
  {
    DiagonalMatrix<LinearAlgebra::distributed::Vector<double,
      MemorySpace::CUDA>> preconditioner;
    preconditioner.get_vector().reinit(system_rhs_dev);
    preconditioner.get_vector() = 1.;

    if(false)
    for (unsigned int i=0; i<10; ++i)
      {
        Timer time;
        IterationNumberControl solver_control(20,
                                              1e-6 * system_rhs_dev.l2_norm());
        SolverCG<LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA>> cg(
                                                                                   solver_control);
        solution_dev = 0;
        cg.solve(*system_matrix_dev, solution_dev, system_rhs_dev, preconditioner);

        cudaDeviceSynchronize();
        pcout << "   Solved in " << solver_control.last_step()
              << " iterations with time " << time.wall_time()
              << " and DoFs/s " << static_cast<double>(dof_handler.n_dofs()) * solver_control.last_step() / time.wall_time()
              << " norm " << solution_dev.l2_norm()
              << std::endl;
      }

    pcout << std::endl;

    for (unsigned int i=0; i<1; ++i)
      {
        Timer time;
        IterationNumberControl solver_control(60,
                                              1e-6 * system_rhs_dev.l2_norm());
        SolverCG2<LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA>> cg(
                                                                                   solver_control);
        solution_dev = 0;
        cg.solve(*system_matrix_dev, solution_dev, system_rhs_dev, preconditioner);

        cudaDeviceSynchronize();
        pcout << "   Solved in " << solver_control.last_step()
              << " iterations with time " << time.wall_time()
              << " and DoFs/s " << static_cast<double>(dof_handler.n_dofs()) * solver_control.last_step() / time.wall_time()
              << " norm " << solution_dev.l2_norm()
              << std::endl;
      }

    pcout << std::endl;

    for (unsigned int i=0; i<10; ++i)
      {
        Timer time;

        for (unsigned int t=0; t<100; ++t)
          system_matrix_dev->vmult(system_rhs_dev, solution_dev);

        cudaDeviceSynchronize();
        pcout << "   100 mat-vecs in time " << time.wall_time()
              << " and DoFs/s " << static_cast<double>(dof_handler.n_dofs()) * 100. / time.wall_time()
              << std::endl;
      }

    LinearAlgebra::ReadWriteVector<double> rw_vector(locally_owned_dofs);
    rw_vector.import(solution_dev, VectorOperation::insert);
    ghost_solution_host.import(rw_vector, VectorOperation::insert);

    constraints.distribute(ghost_solution_host);

    ghost_solution_host.update_ghost_values();
  }

  template <int dim, int fe_degree>
  void HelmholtzProblem<dim, fe_degree>::output_results(
    const unsigned int cycle) const
  {
    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(ghost_solution_host, "solution");
    data_out.build_patches();

    std::ofstream output(
      "solution-" + std::to_string(cycle) + "." +
      std::to_string(Utilities::MPI::this_mpi_process(mpi_communicator)) +
      ".vtu");
    DataOutBase::VtkFlags flags;
    flags.compression_level = DataOutBase::VtkFlags::best_speed;
    data_out.set_flags(flags);
    data_out.write_vtu(output);

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        std::vector<std::string> filenames;
        for (unsigned int i = 0;
             i < Utilities::MPI::n_mpi_processes(mpi_communicator);
             ++i)
          filenames.emplace_back("solution-" + std::to_string(cycle) + "." +
                                 std::to_string(i) + ".vtu");

        std::string master_name =
          "solution-" + Utilities::to_string(cycle) + ".pvtu";
        std::ofstream master_output(master_name);
        data_out.write_pvtu_record(master_output, filenames);
      }

    Vector<float> cellwise_norm(triangulation.n_active_cells());
    VectorTools::integrate_difference(dof_handler,
                                      ghost_solution_host,
                                      Functions::ZeroFunction<dim>(),
                                      cellwise_norm,
                                      QGauss<dim>(fe.degree + 2),
                                      VectorTools::L2_norm);
    const double global_norm =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_norm,
                                        VectorTools::L2_norm);
    pcout << "  solution norm: " << global_norm << std::endl;
  }


  template <int dim, int fe_degree>
  void HelmholtzProblem<dim, fe_degree>::run()
  {
    for (unsigned int cycle = 3; cycle < 4; ++cycle)
      {
        pcout << "Cycle " << cycle << std::endl;

        const unsigned int n_refine = (cycle + 6)/dim;
        const unsigned int remainder = cycle%dim;
        std::vector<unsigned int> subdivisions(dim, 1);
        for (unsigned int d=0; d<remainder; ++d)
          subdivisions[d] = 2;
        Point<dim> p1;
        for (unsigned int d=0; d<dim; ++d)
          p1[d] = -1;
        Point<dim> p2;
        for (unsigned int d=0; d<remainder; ++d)
          p2[d] = 3;
        for (unsigned int d=remainder; d<dim; ++d)
          p2[d] = 1;

        triangulation.clear();
        GridGenerator::subdivided_hyper_rectangle(triangulation, subdivisions,
                                                  p1, p2);
        triangulation.refine_global(n_refine);

        setup_system();

        pcout << "   Number of active cells:       "
              << triangulation.n_global_active_cells() << std::endl
              << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

        assemble_rhs();
        solve();
        output_results(cycle);
        pcout << std::endl;
      }
  }
} // namespace Step64


int main(int argc, char *argv[])
{
  try
    {
      using namespace Step64;

      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

      int         n_devices       = 0;
      cudaError_t cuda_error_code = cudaGetDeviceCount(&n_devices);
      AssertCuda(cuda_error_code);
      std::cout << "Number of CUDA devices: " << n_devices << std::endl;
      const unsigned int my_mpi_id =
        Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
      const int device_id = my_mpi_id % n_devices;
      cuda_error_code     = cudaSetDevice(device_id);
      AssertCuda(cuda_error_code);

      HelmholtzProblem<3, 4> helmholtz_problem;
      helmholtz_problem.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
