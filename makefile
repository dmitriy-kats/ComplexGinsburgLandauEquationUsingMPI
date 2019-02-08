cgl_DK:
	mpicc -c $(ID) cgl_DK.c
	mpicc -o cglMPI $(LD) cgl_DK.o -lfftw3_mpi -lfftw3 -lm
cgl_DK_run:
	mpirun -np 8 cglMPI 8 1.5 0.25 10000

