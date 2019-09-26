#include <mpi.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>

void test2(MPI_Comm comm, const bool is_host)
{
  int my_rank;
  MPI_Comm_rank(comm, &my_rank);



  double *new_val_dev_1;
  double *new_val_dev_2;
  //Utilities::CUDA::malloc(new_val_dev_1, 10);
  if (is_host)
  {
    new_val_dev_1 = (double*)malloc(10 * sizeof(double));
    new_val_dev_2 = (double*)malloc(10 * sizeof(double));
  }
  else
  {
    auto cuda_error_code = cudaMalloc((void**)&new_val_dev_1, 10 * sizeof(double));
    cuda_error_code = cudaMalloc((void**)&new_val_dev_2, 10 * sizeof(double));
  }

  std::cout << "Allocation successful on rank " << my_rank << std::endl;

  if(my_rank == 0)
  {
    MPI_Request requests[2];
    MPI_Isend(new_val_dev_1, 10, MPI_DOUBLE, 1, 0, comm, requests + 0);
    MPI_Irecv(new_val_dev_2, 10, MPI_DOUBLE, 1, 0, comm, requests + 1);

    MPI_Status statuses[2];
    MPI_Waitall(2, requests, statuses);
  }
  else
  {
    MPI_Request requests[2];
    MPI_Isend(new_val_dev_1, 10, MPI_DOUBLE, 0, 0, comm, requests + 0);
    MPI_Irecv(new_val_dev_2, 10, MPI_DOUBLE, 0, 0, comm, requests + 1);

    MPI_Status statuses[2];
    MPI_Waitall(2, requests, statuses);
  }
  std::cout << "Send successful" << my_rank << std::endl;
} 

int main(int argc, char**argv)
{
  MPI_Init(&argc, &argv);

  test2(MPI_COMM_WORLD, true);
  test2(MPI_COMM_WORLD, false);

  MPI_Finalize();
}
