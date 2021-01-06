/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2019-2021 by the deal.II authors
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
#include <deal.II/base/cuda.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/revision.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/matrix_free/cuda_fe_evaluation.h>
#include <deal.II/matrix_free/cuda_matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <limits>

#include "fe_evaluation_gl.h"

#define MERGED_COEFFICIENTS
//#define COLLOCATION 
#define OPTIMIZED_UPDATE

#include "solver.h"

namespace BP5
{
  using namespace dealii;



  // Variant with merged coefficient tensor
  template <int dim, int fe_degree>
  class JacobianFunctor
  {
  public:
    JacobianFunctor(double *coefficient, const unsigned int n_cells)
      : coef(coefficient)
      , n_cells(n_cells)
    {}

    __device__ void
    operator()(
      const unsigned int                                          cell,
      const typename CUDAWrappers::MatrixFree<dim, double>::Data *gpu_data);

    static const unsigned int n_dofs_1d  = fe_degree + 1;
    static const unsigned int n_q_points = Utilities::pow(n_dofs_1d, dim);

  private:
    double *           coef;
    const unsigned int n_cells;
  };



  template <int dim, int fe_degree>
  __device__ void
  JacobianFunctor<dim, fe_degree>::operator()(
    const unsigned int                                          cell,
    const typename CUDAWrappers::MatrixFree<dim, double>::Data *gpu_data)
  {
    const unsigned int q = CUDAWrappers::q_point_id_in_cell<dim>(fe_degree + 1);
    Tensor<2, dim>     inv_jac;
    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int e = 0; e < dim; ++e)
        inv_jac[d][e] =
          gpu_data->inv_jacobian[cell * gpu_data->padding_length + q +
                                 gpu_data->n_cells * gpu_data->padding_length *
                                   (d * dim + e)];
    Tensor<2, dim> my_coef;
    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int e = d; e < dim; ++e)
        {
          double sum = inv_jac[d][0] * inv_jac[e][0];
          for (unsigned int f = 1; f < dim; ++f)
            sum += inv_jac[d][f] * inv_jac[e][f];
          my_coef[d][e] = sum;
        }
    for (unsigned int d = 0; d < dim; ++d)
      coef[q + cell * n_q_points + d * n_cells * n_q_points] =
        gpu_data->JxW[gpu_data->padding_length * cell + q] * my_coef[d][d];
    for (unsigned int c = dim, d = 0; d < dim; ++d)
      for (unsigned int e = d + 1; e < dim; ++e, ++c)
        coef[q + cell * n_q_points + c * n_cells * n_q_points] =
          gpu_data->JxW[gpu_data->padding_length * cell + q] * my_coef[d][e];
  }



  template <int dim, int fe_degree>
  class LocalPoissonOperator
  {
  public:
    LocalPoissonOperator(double *           coefficient,
                                 const unsigned int n_cells)
      : coef(coefficient)
      , n_cells(n_cells)
    {}

    __device__ void
    operator()(
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
    double *           coef;
  };



  template <int dim, int fe_degree>
  __device__ void
  LocalPoissonOperator<dim, fe_degree>::operator()(
    const unsigned int                                          cell,
    const typename CUDAWrappers::MatrixFree<dim, double>::Data *gpu_data,
    CUDAWrappers::SharedData<dim, double> *                     shared_data,
    const double *                                              src,
    double *                                                    dst) const
  {
    CUDAWrappers::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double>
      fe_eval(cell, gpu_data, shared_data);
    fe_eval.read_dof_values(src);
    fe_eval.evaluate(false, true);
#ifdef MERGED_COEFFICIENTS
    const unsigned int offset = n_q_points * n_cells;
    const unsigned int q =
      CUDAWrappers::internal::compute_index<dim, fe_degree + 1>();
    if (dim == 3)
      {
        const double grad0           = shared_data->gradients[0][q];
        const double grad1           = shared_data->gradients[1][q];
        const double grad2           = shared_data->gradients[2][q];
        shared_data->gradients[0][q] = grad0 * coef[q] +
                                       grad1 * coef[q + 3 * offset] +
                                       grad2 * coef[q + 4 * offset];
        shared_data->gradients[1][q] = grad0 * coef[q + 3 * offset] +
                                       grad1 * coef[q + 1 * offset] +
                                       grad2 * coef[q + 5 * offset];
        shared_data->gradients[2][q] = grad0 * coef[q + 4 * offset] +
                                       grad1 * coef[q + 5 * offset] +
                                       grad2 * coef[q + 2 * offset];
      }
    else
      {
        const double grad0 = shared_data->gradients[0][q];
        const double grad1 = shared_data->gradients[1][q];
        shared_data->gradients[0][q] =
          grad0 * coef[q] + grad1 * coef[q + 2 * offset];
        shared_data->gradients[1][q] =
          grad0 * coef[q + 2 * offset] + grad1 * coef[q + 1 * offset];
      }
    __syncthreads();
#else
    fe_eval.submit_gradient(fe_eval.get_gradient());
#endif
    fe_eval.integrate(false, true);
    fe_eval.distribute_local_to_global(dst);
  }



  template <int dim, int fe_degree>
  class PoissonOperator
  {
  public:
    PoissonOperator(const DoFHandler<dim> &          dof_handler,
                    const AffineConstraints<double> &constraints);

    void
    vmult(LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> &dst,
          const LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA>
            &src) const;

    void
    initialize_dof_vector(
      LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> &vec) const
    {
      mf_data.initialize_dof_vector(vec);
    }

  private:
    CUDAWrappers::MatrixFree<dim, double>       mf_data;
    LinearAlgebra::CUDAWrappers::Vector<double> coef;
    unsigned int                                n_owned_cells;

  public:
    bool do_zero_out;
  };



  template <int dim, int fe_degree>
  PoissonOperator<dim, fe_degree>::PoissonOperator(
    const DoFHandler<dim> &          dof_handler,
    const AffineConstraints<double> &constraints)
    : do_zero_out(true)
  {
    MappingQGeneric<dim> mapping(fe_degree);
    typename CUDAWrappers::MatrixFree<dim, double>::AdditionalData
      additional_data;
    additional_data.mapping_update_flags = update_values | update_gradients |
                                           update_JxW_values |
                                           update_quadrature_points;

    additional_data.overlap_communication_computation = true;

#ifdef COLLOCATION
    const QGaussLobatto<1> quad(fe_degree + 1);
#else
    const QGauss<1> quad(fe_degree + 1);
#endif
    mf_data.reinit(mapping, dof_handler, constraints, quad, additional_data);

    n_owned_cells = dynamic_cast<const parallel::Triangulation<dim> *>(
                      &dof_handler.get_triangulation())
                      ->n_locally_owned_active_cells();
    coef.reinit(Utilities::pow(fe_degree + 1, dim) * n_owned_cells * dim *
                (dim + 1) / 2);

    const JacobianFunctor<dim, fe_degree> functor(coef.get_values(),
                                                  n_owned_cells);
    mf_data.evaluate_coefficients(functor);
  }



  template <int dim, int fe_degree>
  void
  PoissonOperator<dim, fe_degree>::vmult(
    LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> &      dst,
    const LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> &src)
    const
  {
    if (do_zero_out)
      dst = 0.;
    LocalPoissonOperator<dim, fe_degree> local_poisson_operator(
      coef.get_values(), n_owned_cells);
    mf_data.cell_loop(local_poisson_operator, src, dst);
    mf_data.copy_constrained_values(src, dst);
  }



  template <int dim, int fe_degree>
  class PoissonProblem
  {
  public:
    PoissonProblem();

    void
    run(unsigned int cycle_min,
        unsigned int cycle_max,
        unsigned int n_iterations,
        unsigned int n_repetitions,
        unsigned int min_run);

  private:
    void
    setup_system();

    void
    assemble_rhs();

    void
    solve(unsigned int n_iterations,
          unsigned int n_repetitions,
          unsigned int min_run);

    void
    output_results(const unsigned int cycle) const;

    MPI_Comm mpi_communicator;

    parallel::distributed::Triangulation<dim> triangulation;

    FE_Q<dim>       fe;
    DoFHandler<dim> dof_handler;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    AffineConstraints<double>                        constraints;
    std::unique_ptr<PoissonOperator<dim, fe_degree>> system_matrix_dev;

    LinearAlgebra::distributed::Vector<double, MemorySpace::Host> ghost_solution_host;
    LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> solution_dev;
    LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> system_rhs_dev;

    ConditionalOStream pcout;
  };



  template <int dim, int fe_degree>
  PoissonProblem<dim, fe_degree>::PoissonProblem()
    : mpi_communicator(MPI_COMM_WORLD)
    , triangulation(mpi_communicator)
    , fe(fe_degree)
    , dof_handler(triangulation)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
  {}



  template <int dim, int fe_degree>
  void
  PoissonProblem<dim, fe_degree>::setup_system()
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
      new PoissonOperator<dim, fe_degree>(dof_handler, constraints));

    ghost_solution_host.reinit(locally_owned_dofs,
                               locally_relevant_dofs,
                               mpi_communicator);
    system_matrix_dev->initialize_dof_vector(solution_dev);
    system_rhs_dev.reinit(solution_dev);
  }



  template <int dim, int fe_degree>
  void
  PoissonProblem<dim, fe_degree>::assemble_rhs()
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
  void
  PoissonProblem<dim, fe_degree>::solve(unsigned int n_iterations,
                                        unsigned int n_repetitions,
                                        unsigned int min_run)
  {
    DiagonalMatrix<
      LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA>>
      preconditioner;
    preconditioner.get_vector().reinit(system_rhs_dev);
    preconditioner.get_vector() = 1.;

    if (min_run == 0) // solve with SolverCG
      {
        double throughput_max = std::numeric_limits<double>::min();

        for (unsigned int i = 0; i < n_repetitions; ++i)
          {
            system_matrix_dev->do_zero_out = true;

            Timer                  time;
            IterationNumberControl solver_control(n_iterations,
                                                  1e-6 *
                                                    system_rhs_dev.l2_norm());
            SolverCG<
              LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA>>
              cg(solver_control);
            solution_dev = 0;
            cg.solve(*system_matrix_dev,
                     solution_dev,
                     system_rhs_dev,
                     preconditioner);

            cudaDeviceSynchronize();

            const double measured_time = time.wall_time();
            const double measured_throughput =
              static_cast<double>(dof_handler.n_dofs()) *
              solver_control.last_step() / measured_time /
              Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

            throughput_max = std::max(throughput_max, measured_throughput);

            pcout << "   Solved in " << solver_control.last_step()
                  << " iterations with time " << measured_time << " and DoFs/s "
                  << measured_throughput << " norm " << solution_dev.l2_norm()
                  << std::endl;
          }
        pcout << "pcg-standard "
              << dof_handler.n_dofs() /
                   Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
              << " " << throughput_max << std::endl
              << std::endl;
      }

    if(true) // solve with optimized SolverCG
      {
        double throughput_max = std::numeric_limits<double>::min();
  
        for (unsigned int i = 0; i < n_repetitions; ++i)
          {
            system_matrix_dev->do_zero_out = false;
            Timer                  time;
            IterationNumberControl solver_control(n_iterations,
                                                  1e-6 *
                                                    system_rhs_dev.l2_norm());
            SolverCGFullMerge<
              LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA>>
              cg(solver_control);
            solution_dev = 0;
            cg.solve(*system_matrix_dev,
                     solution_dev,
                     system_rhs_dev,
                     preconditioner);
  
            cudaDeviceSynchronize();
  
            const double measured_time = time.wall_time();
            const double measured_throughput =
              static_cast<double>(dof_handler.n_dofs()) *
              solver_control.last_step() / measured_time /
              Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  
            throughput_max = std::max(throughput_max, measured_throughput);
  
            pcout << "   Solved in " << solver_control.last_step()
                  << " iterations with time " << measured_time << " and DoFs/s "
                  << measured_throughput << " norm " << solution_dev.l2_norm()
                  << std::endl;
          }
        pcout << "pcg-merged "
              << dof_handler.n_dofs() /
                   Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
              << " " << throughput_max << std::endl
              << std::endl;
      }

    if (min_run == 0) // vmult
      {
        double throughput_max = std::numeric_limits<double>::min();

        for (unsigned int i = 0; i < n_repetitions; ++i)
          {
            Timer time;

            for (unsigned int t = 0; t < n_iterations; ++t)
              system_matrix_dev->vmult(system_rhs_dev, solution_dev);

            cudaDeviceSynchronize();

            const double measured_time = time.wall_time();
            const double measured_throughput =
              static_cast<double>(dof_handler.n_dofs()) * n_iterations /
              measured_time / Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

            throughput_max = std::max(throughput_max, measured_throughput);

            pcout << "   " << n_iterations << " mat-vecs in time "
                  << measured_time << " and DoFs/s " << measured_throughput
                  << std::endl;
          }
        pcout << "vmult "
              << dof_handler.n_dofs() /
                   Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
              << " " << throughput_max << std::endl
              << std::endl;
      }


    if (min_run == 0) // Paraview output, computation of error, ...
      {
        LinearAlgebra::ReadWriteVector<double> rw_vector(locally_owned_dofs);
        rw_vector.import(solution_dev, VectorOperation::insert);
        ghost_solution_host.import(rw_vector, VectorOperation::insert);

        constraints.distribute(ghost_solution_host);

        ghost_solution_host.update_ghost_values();
      }
  }



  template <int dim, int fe_degree>
  void
  PoissonProblem<dim, fe_degree>::output_results(const unsigned int cycle) const
  {
    if(false)
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
      }

    if(true)
      {
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
  }



  template <int dim, int fe_degree>
  void
  PoissonProblem<dim, fe_degree>::run(unsigned int cycle_min,
                                      unsigned int cycle_max,
                                      unsigned int n_iterations,
                                      unsigned int n_repetitions,
                                      unsigned int min_run)
  {
    for (unsigned int cycle = cycle_min; cycle <= cycle_max; ++cycle)
      {
        pcout << "Cycle " << cycle << std::endl;

        unsigned int       n_refine  = cycle / 6;
        const unsigned int remainder = cycle % 6;

        std::vector<unsigned int>                 subdivisions(dim, 1);
        if (remainder == 1 && cycle > 1)
          {
            subdivisions[0] = 3;
            subdivisions[1] = 2;
            subdivisions[2] = 2;
            n_refine -= 1;
          }
        if (remainder == 2)
          subdivisions[0] = 2;
        else if (remainder == 3)
          subdivisions[0] = 3;
        else if (remainder == 4)
          subdivisions[0] = subdivisions[1] = 2;
        else if (remainder == 5)
          {
            subdivisions[0] = 3;
            subdivisions[1] = 2;
          }

        Point<dim> p2;
        for (unsigned int d = 0; d < dim; ++d)
          p2[d] = subdivisions[d];

        triangulation.clear();
        GridGenerator::subdivided_hyper_rectangle(triangulation, subdivisions, Point<dim>(), p2);

        triangulation.refine_global(n_refine);

        setup_system();

        pcout << "   Number of active cells:       "
              << triangulation.n_global_active_cells() << std::endl
              << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl
              << std::endl;

        assemble_rhs();
        solve(n_iterations, n_repetitions, min_run);
        output_results(cycle);
        pcout << std::endl;
      }
  }
} // namespace BP5



void
print_hardware_specs()
{
  using namespace dealii;

  ConditionalOStream pcout(
    std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  pcout << std::endl
        << "deal.II info:" << std::endl
        << std::endl
        << "  deal.II git version " << DEAL_II_GIT_SHORTREV << " on branch "
        << DEAL_II_GIT_BRANCH << std::endl
        << "  with vectorization level = "
        << DEAL_II_COMPILER_VECTORIZATION_LEVEL << std::endl;

  int         n_devices       = 0;
  cudaError_t cuda_error_code = cudaGetDeviceCount(&n_devices);
  AssertCuda(cuda_error_code);
  pcout << "  number of CUDA devices = " << n_devices << std::endl
        << std::endl;
  const unsigned int my_mpi_id =
    Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const int device_id = my_mpi_id % n_devices;
  cuda_error_code     = cudaSetDevice(device_id);
  AssertCuda(cuda_error_code);
}



int
main(int argc, char *argv[])
{
  try
    {
      using namespace BP5;

      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

      print_hardware_specs();

      const unsigned int dim           = 3;
      const unsigned int degree        = 4;
      const unsigned int min_run       = 0;
      const unsigned int cycle_min     = 7;
      const unsigned int cycle_max     = 40;
      const unsigned int n_iterations  = 200;
      const unsigned int n_repetitions = 10;

      PoissonProblem<dim, degree> poisson_problem;
      poisson_problem.run(cycle_min, cycle_max, n_iterations, n_repetitions, min_run);
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
