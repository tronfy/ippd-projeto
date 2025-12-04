gerador:
	gcc -o gerador_dataset gerador_dataset.c -O3

dataset:
	./gerador_dataset 3000 5 1000 debug_data.txt

seq:
	gcc -o kmeans_sequencial kmeans_sequencial.c -Ofast

omp:
	gcc -o kmeans_openmp kmeans_openmp.c -fopenmp -Ofast

pth:
	gcc -o kmeans_pthreads kmeans_pthreads.c -lpthread -mavx2 -Ofast

mpi:
	mpicc -o kmeans_mpi kmeans_mpi.c -mavx2 -Ofast -floop-optimize -fprefetch-loop-arrays
