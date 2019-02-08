/*  Complex Ginsburg-Landau Equation for ESAM_444 Class
  Program written by Dmitriy Kats on May 23, 2018

This program solves the complex Ginsburg-Landau equation on the domain that is L=128(pi) on each side.
  Periodic boundary conditions are used.
  A Pseudo-spectral method with fourth order Runge Kutta is used. 
  The initial data is the range [-1.5, 1.5]+i[-1.5, 1.5]
  The run time is until time is 10^4

  Inputs: N is the number of grids in each direction
  	  	  c1 and c3 are the real valued coefficients
  	  	  M is the number of time steps 
  	  	  seed is an optional input parameter

  Outputs: The code outputs the grid values of A in a file called CGL.out

 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#include "fftw3-mpi.h"


int main(int argc, char* argv[])
{
	//Variables used for timing
	double precision = MPI_Wtick();
	double starttime = MPI_Wtime();

	//Intialize MPI and fftw
	MPI_Init(&argc, &argv);
	fftw_mpi_init();

	//find the rank of the processor
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	// Determine the number of processes
	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// I could have set up temporary int values to take N and M and then broadcasted them as I did
	//with the other variables. It felt clunky.
	const ptrdiff_t N=atoi(argv[1]);
	const ptrdiff_t M=atoi(argv[4]);
	double c1, c3;
	long int seed;

	if(rank==0)
	{
		//Required inputs are below
		c1 = atof(argv[2]);
		c3 = atof(argv[3]);
		//set seed
		if(argc!=4)
		{
			FILE* urand = fopen ( "/dev/urandom","r");
			fread(&seed,sizeof(long int),1,urand);
			fclose(urand);
		}
		else
		{
			seed = atol(argv[5]);
		}

	}
	//Broadcast all the inputs to the other processors 
	MPI_Bcast(&c1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&c3, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&seed, 1, MPI_LONG_INT, 0, MPI_COMM_WORLD);

	seed+=rank; //add rank so all the processors have different seeds
	srand48(seed);

	ptrdiff_t localM, local0;
	//time step and coefficients used in calcualtions
	double dt = 10000.0/((double) M); 
	double sqFac = 4.0/(128.0*128.0);
	double dtover4= dt/4.0;
	double dtover3= dt/3.0;
	double dtover2= dt/2.0;


	//This array will store the multipliers for the laplace operator
	double fftmultipliers [N]; 

	int i, j, tt;

	for (i=0; i<N/2+1; i++)
	{ //determine fft multipliers for the first half;
		// the multipliers include the division by N^2
		fftmultipliers[i]=-(((double)N)/2.0-N/2+i)*(((double)N)/2.0-N/2+i)/(((double)N)*((double)N));
	}

	for (i=0; i<N/2; i++)
	{//mirror the fft multipliers for the second half
		fftmultipliers[N/2+1+i]=fftmultipliers[N/2-1-i];
	}


	//figure out size of local arrays
	ptrdiff_t alloc_local = fftw_mpi_local_size_2d(N,N,MPI_COMM_WORLD, &localM, &local0);

	//All the variables of A are just the A data at different points of the calcualtion
	fftw_complex* A = fftw_alloc_complex(alloc_local);
	fftw_complex* d2A = fftw_alloc_complex(alloc_local); //Laplace operator
	fftw_complex* A1 = fftw_alloc_complex(alloc_local);
	fftw_complex* A2 = fftw_alloc_complex(alloc_local);
	fftw_complex* Ao = fftw_alloc_complex(alloc_local); //Stores original A data

	fftw_plan pf, pb;
	//plans for forward and backward
	pf = fftw_mpi_plan_dft_2d(N,N, A, d2A, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);
	pb = fftw_mpi_plan_dft_2d(N,N, d2A, d2A, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);

	//Intialize values to be random numbers between -1.5 and 1.5
	for (i=0; i<localM; ++i)
	{
		for(j=0; j<N; ++j)
		{
			A[i*N+j][0]=3.0*drand48()-1.5;
			A[i*N+j][1]=3.0*drand48()-1.5;
		}
	}

	double* fulldata = NULL; //stores the full data for later writing

	if (rank==0)
	{
		fulldata=(double*)malloc(2*N*N*sizeof(double));
	}

	//Write original data to "A.out"
	MPI_Gather(A, 2*localM*N, MPI_DOUBLE, fulldata, 2*localM*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	if (rank==0){
		FILE *fp0 = fopen("A.out", "w");
		fwrite(fulldata, sizeof(double), 2*N*N, fp0);
		fclose (fp0);}

	MPI_Gather(A, 2*localM*N, MPI_DOUBLE, fulldata, 2*localM*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	if (rank==0)
	{FILE *fp = fopen("CGL.out", "a");
	fwrite(fulldata, sizeof(double), 2*N*N, fp);
	fclose (fp);
	//printf("Appending data at t=%f\n", tt*dt);
	}



	for(tt=0; tt<M; tt++) //time loop for one step for debugging
	{


		//Calculate the d2A(A)
		fftw_execute(pf); //perform FFT and now the transformed data is in d2A
		for (i=0; i<localM; ++i) //calcualte the laplacian and it is stored in d2A
		{
			for(j=0; j<N; ++j)
			{	//multipliers are for the coefficients in x and y directions as well as the division by N^2
				d2A[i*N+j][0]=d2A[i*N+j][0]*(fftmultipliers[j]+fftmultipliers[local0+i]);
				d2A[i*N+j][1]=d2A[i*N+j][1]*(fftmultipliers[j]+fftmultipliers[local0+i]);
			}
		}
		fftw_execute(pb); //perform IFFT and Laplace data is still in d2A. Original data in A.

		//calcualte A1 and put it into A for the FFT plans; also store A data in Ao
		for (i=0; i<localM; ++i) 
		{
			for(j=0; j<N; ++j)
			{	//I wrote all the multiplication by hand to make sure the complex mulitplication was right and 
				//then programmed it below
				//Update real part
				A1[i*N+j][0]=A[i*N+j][0]+
						dtover4*(A[i*N+j][0]+sqFac*(d2A[i*N+j][0]-c1*d2A[i*N+j][1])-
								(A[i*N+j][0]*A[i*N+j][0]*A[i*N+j][0]+A[i*N+j][0]*A[i*N+j][1]*A[i*N+j][1]
								                                                                      +c3*A[i*N+j][0]*A[i*N+j][0]*A[i*N+j][1]+c3*A[i*N+j][1]*A[i*N+j][1]*A[i*N+j][1]));
				//Update imaginary part
				A1[i*N+j][1]=A[i*N+j][1]+
						dtover4*(A[i*N+j][1]+sqFac*(d2A[i*N+j][1]+c1*d2A[i*N+j][0])-
								(-c3*A[i*N+j][0]*A[i*N+j][0]*A[i*N+j][0]-c3*A[i*N+j][1]*A[i*N+j][1]*A[i*N+j][0]+
										A[i*N+j][0]*A[i*N+j][0]*A[i*N+j][1]+A[i*N+j][1]*A[i*N+j][1]*A[i*N+j][1]));
				
				Ao[i*N+j][0]=A[i*N+j][0];
				Ao[i*N+j][1]=A[i*N+j][1];
			}
		}

		for (i=0; i<localM; ++i) 
		{
			for(j=0; j<N; ++j)
			{
				A[i*N+j][0]=A1[i*N+j][0];
				A[i*N+j][1]=A1[i*N+j][1];
			}
		}

		//Calculate the d2A(A1);  NOTE:the A1 data is actually called A now for the FFT plan to work
		fftw_execute(pf); //perform FFT and now the transformed data is in d2A
		for (i=0; i<localM; ++i) //calcualte the laplacian and it is stored in d2A
		{
			for(j=0; j<N; ++j)
			{	//multipliers are for the coefficients in x and y directions. 
				//Divide by the number of grid points since FFTW doesn't do it automatically.
				d2A[i*N+j][0]=d2A[i*N+j][0]*(fftmultipliers[j]+fftmultipliers[local0+i]);
				d2A[i*N+j][1]=d2A[i*N+j][1]*(fftmultipliers[j]+fftmultipliers[local0+i]);
			}
		}
		fftw_execute(pb); //perform IFFT and Laplace data is still in d2A. Original data in Ao.


		//calcualte A2 and put it into A for the FFT plans
		for (i=0; i<localM; ++i) 
		{
			for(j=0; j<N; ++j)
			{	
				A2[i*N+j][0]=Ao[i*N+j][0]+
						dtover3*(A[i*N+j][0]+sqFac*(d2A[i*N+j][0]-c1*d2A[i*N+j][1])-
								(A[i*N+j][0]*A[i*N+j][0]*A[i*N+j][0]+A[i*N+j][0]*A[i*N+j][1]*A[i*N+j][1]
								                                                                      +c3*A[i*N+j][0]*A[i*N+j][0]*A[i*N+j][1]+c3*A[i*N+j][1]*A[i*N+j][1]*A[i*N+j][1]));
				A2[i*N+j][1]=Ao[i*N+j][1]+
						dtover3*(A[i*N+j][1]+sqFac*(d2A[i*N+j][1]+c1*d2A[i*N+j][0])-
								(-c3*A[i*N+j][0]*A[i*N+j][0]*A[i*N+j][0]-c3*A[i*N+j][1]*A[i*N+j][1]*A[i*N+j][0]+
										A[i*N+j][0]*A[i*N+j][0]*A[i*N+j][1]+A[i*N+j][1]*A[i*N+j][1]*A[i*N+j][1]));

			}
		}

		for (i=0; i<localM; ++i) 
		{
			for(j=0; j<N; ++j)
			{
				A[i*N+j][0]=A2[i*N+j][0];
				A[i*N+j][1]=A2[i*N+j][1];

			}
		}

		//Calculate the d2A(A2);  NOTE:the A2 data is actually called A now for the FFT plan to work
		fftw_execute(pf); //perform FFT and now the transformed data is in d2A
		for (i=0; i<localM; ++i) //calcualte the laplacian and it is stored in d2A
		{
			for(j=0; j<N; ++j)
			{	//multipliers are for the coefficients in x and y directions. 
				//Divide by the number of grid points since FFTW doesn't do it automatically.
				d2A[i*N+j][0]=d2A[i*N+j][0]*(fftmultipliers[j]+fftmultipliers[local0+i]);
				d2A[i*N+j][1]=d2A[i*N+j][1]*(fftmultipliers[j]+fftmultipliers[local0+i]);
			}
		}
		fftw_execute(pb); //perform IFFT and Laplace data is still in d2A.

		//calcualte A1 and put it into A for the FFT plans
		for (i=0; i<localM; ++i) 
		{
			for(j=0; j<N; ++j)
			{	
				A1[i*N+j][0]=Ao[i*N+j][0]+
						dtover2*(A[i*N+j][0]+sqFac*(d2A[i*N+j][0]-c1*d2A[i*N+j][1])-
								(A[i*N+j][0]*A[i*N+j][0]*A[i*N+j][0]+A[i*N+j][0]*A[i*N+j][1]*A[i*N+j][1]
								                                                                      +c3*A[i*N+j][0]*A[i*N+j][0]*A[i*N+j][1]+c3*A[i*N+j][1]*A[i*N+j][1]*A[i*N+j][1]));
				A1[i*N+j][1]=Ao[i*N+j][1]+
						dtover2*(A[i*N+j][1]+sqFac*(d2A[i*N+j][1]+c1*d2A[i*N+j][0])-
								(-c3*A[i*N+j][0]*A[i*N+j][0]*A[i*N+j][0]-c3*A[i*N+j][1]*A[i*N+j][1]*A[i*N+j][0]+
										A[i*N+j][0]*A[i*N+j][0]*A[i*N+j][1]+A[i*N+j][1]*A[i*N+j][1]*A[i*N+j][1]));

			}
		}

		for (i=0; i<localM; ++i) 
		{
			for(j=0; j<N; ++j)
			{
				A[i*N+j][0]=A1[i*N+j][0];
				A[i*N+j][1]=A1[i*N+j][1];
			}
		}

		//Calculate the d2A(A1);  NOTE:the A1 data is actually called A now for the FFT plan to work
		fftw_execute(pf); //perform FFT and now the transformed data is in d2A
		for (i=0; i<localM; ++i) //calcualte the laplacian and it is stored in d2A
		{
			for(j=0; j<N; ++j)
			{	//multipliers are for the coefficients in x and y directions. 
				//Divide by the number of grid points since FFTW doesn't do it automatically.
				d2A[i*N+j][0]=d2A[i*N+j][0]*(fftmultipliers[j]+fftmultipliers[local0+i]);
				d2A[i*N+j][1]=d2A[i*N+j][1]*(fftmultipliers[j]+fftmultipliers[local0+i]);
			}
		}
		fftw_execute(pb); //perform IFFT and Laplace data is still in d2A.


		//calcualte A. It is stored temporarily in A1 and then moved to A. 
		for (i=0; i<localM; ++i) 
		{
			for(j=0; j<N; ++j)
			{	
				A1[i*N+j][0]=Ao[i*N+j][0]+
						dt*(A[i*N+j][0]+sqFac*(d2A[i*N+j][0]-c1*d2A[i*N+j][1])-
								(A[i*N+j][0]*A[i*N+j][0]*A[i*N+j][0]+A[i*N+j][0]*A[i*N+j][1]*A[i*N+j][1]
								                                                                      +c3*A[i*N+j][0]*A[i*N+j][0]*A[i*N+j][1]+c3*A[i*N+j][1]*A[i*N+j][1]*A[i*N+j][1]));
				A1[i*N+j][1]=Ao[i*N+j][1]+
						dt*(A[i*N+j][1]+sqFac*(d2A[i*N+j][1]+c1*d2A[i*N+j][0])-
								(-c3*A[i*N+j][0]*A[i*N+j][0]*A[i*N+j][0]-c3*A[i*N+j][1]*A[i*N+j][1]*A[i*N+j][0]+
										A[i*N+j][0]*A[i*N+j][0]*A[i*N+j][1]+A[i*N+j][1]*A[i*N+j][1]*A[i*N+j][1]));

			}
		}
		for (i=0; i<localM; ++i) 
		{
			for(j=0; j<N; ++j)
			{	
				A[i*N+j][0]=A1[i*N+j][0];
				A[i*N+j][1]=A1[i*N+j][1];

			}
		}



		if ((tt+1)%(M/10)==0)
		{//Gather all the data into fulldata
			MPI_Gather(A, 2*localM*N, MPI_DOUBLE, fulldata, 2*localM*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			if (rank==0)
			{FILE *fp = fopen("CGL.out", "a");
			fwrite(fulldata, sizeof(double), 2*N*N, fp);
			fclose (fp);
			//printf("Appending data at t=%f\n", (tt+1)*dt);
			}


		}


	}


	//destroy plans
	fftw_destroy_plan(pf);
	fftw_destroy_plan(pb);
	//free the arrays
	fftw_free(A);
	fftw_free(d2A);
	fftw_free(A1);
	fftw_free(A2);
	fftw_free(Ao);

	if(rank==0)
	{ //free fulldata
		free(fulldata);
	}


	double code_time_in_seconds=MPI_Wtime()-starttime;

	if(rank==0)
	{
		printf("Using %i processors for N=%i. Rank 0 took %le seconds\n\n", size, N, rank, code_time_in_seconds);
	}


	MPI_Finalize();



}
