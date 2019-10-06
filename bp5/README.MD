```bash
srun -N 1 ./step-64 1 1 200 3 12 | tee result_1.txt
srun -N 2 ./step-64 1 1 200 3 13 | tee result_2.txt
srun -N 4 ./step-64 1 1 200 3 14 | tee result_4.txt
```

```bash
srun -N  1 ./step-64 1 1 200 12 | tee result_strong.txt
srun -N  2 ./step-64 1 1 200 12 | tee -a result_strong.txt
srun -N  4 ./step-64 1 1 200 12 | tee -a result_strong.txt
srun -N  8 ./step-64 1 1 200 12 | tee -a result_strong.txt
srun -N 16 ./step-64 1 1 200 12 | tee -a result_strong.txt
srun -N 32 ./step-64 1 1 200 12 | tee -a result_strong.txt
srun -N 60 ./step-64 1 1 200 12 | tee -a result_strong.txt
```

```bash
srun -N  1 ./step-64 1 1 200 10 | tee result_weak.txt
srun -N  2 ./step-64 1 1 200 11 | tee -a result_weak.txt
srun -N  4 ./step-64 1 1 200 12 | tee -a result_weak.txt
srun -N  8 ./step-64 1 1 200 13 | tee -a result_weak.txt
srun -N 16 ./step-64 1 1 200 14 | tee -a result_weak.txt
srun -N 32 ./step-64 1 1 200 15 | tee -a result_weak.txt
srun -N 60 ./step-64 1 1 200 16 | tee -a result_weak.txt
```
