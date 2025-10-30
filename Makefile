gerador:
	gcc -o gerador_dataset gerador_dataset.c -O3

dataset:
	./gerador_dataset 3000 5 1000 debug_data.txt

seq:
	gcc -o kmeans_sequencial kmeans_sequencial.c -O3

omp:
	gcc -o kmeans_openmp kmeans_openmp.c -fopenmp -O3

pth:
	gcc -o kmeans_pthreads kmeans_pthreads.c -lpthread -O3

mpi:
	mpicc -o kmeans_mpi kmeans_mpi.c -O3
