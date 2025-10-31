#define _POSIX_C_SOURCE 199309L // Necessário para CLOCK_MONOTONIC
#include <limits.h>             // Para LLONG_MAX
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h> // Header correto para clock_gettime e struct timespec

// Estrutura para representar um ponto no espaço D-dimensional
typedef struct {
  int *coords;    // Vetor de coordenadas inteiras
  int cluster_id; // ID do cluster ao qual o ponto pertence
} Point;

// --- Funções Utilitárias ---

/**
 * @brief Calcula a distância Euclidiana ao quadrado entre dois pontos com
 * coordenadas inteiras. Usa 'long long' para evitar overflow no cálculo da
 * distância e da diferença.
 * @return A distância Euclidiana ao quadrado como um long long.
 */
long long euclidean_dist_sq(Point *p1, Point *p2, int D) {
  long long dist = 0;
  for (int i = 0; i < D; i++) {
    long long diff = (long long)p1->coords[i] - p2->coords[i];
    dist += diff * diff;
  }
  return dist;
}

// --- Funções Principais do K-Means ---

/**
 * @brief Lê os dados de pontos (inteiros) de um arquivo de texto.
 */
void read_data_from_file(const char *filename, Point *points, int M, int D) {
  FILE *file = fopen(filename, "r");
  if (file == NULL) {
    fprintf(stderr, "Erro: Não foi possível abrir o arquivo '%s'\n", filename);
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < D; j++) {
      if (fscanf(file, "%d", &points[i].coords[j]) != 1) {
        fprintf(stderr,
                "Erro: Arquivo de dados mal formatado ou incompleto.\n");
        fclose(file);
        exit(EXIT_FAILURE);
      }
    }
  }

  fclose(file);
}

/**
 * @brief Inicializa os centroides escolhendo K pontos aleatórios do dataset.
 */
void initialize_centroids(Point *points, Point *centroids, int M, int K,
                          int D) {
  srand(42); // Semente fixa para reprodutibilidade

  int *indices = (int *)malloc(M * sizeof(int));
  for (int i = 0; i < M; i++) {
    indices[i] = i;
  }

  for (int i = 0; i < M; i++) {
    int j = rand() % M;
    int temp = indices[i];
    indices[i] = indices[j];
    indices[j] = temp;
  }

  for (int i = 0; i < K; i++) {
    memcpy(centroids[i].coords, points[indices[i]].coords, D * sizeof(int));
  }

  free(indices);
}

/**
 * @brief Fase de Atribuição: Associa cada ponto ao cluster do centroide mais
 * próximo.
 */
void assign_points_to_clusters(Point *points, Point *centroids, int M, int K,
                               int D) {
  for (int i = 0; i < M; i++) {
    long long min_dist = LLONG_MAX;
    int best_cluster = -1;

    for (int j = 0; j < K; j++) {
      long long dist = euclidean_dist_sq(&points[i], &centroids[j], D);
      if (dist < min_dist) {
        min_dist = dist;
        best_cluster = j;
      }
    }
    points[i].cluster_id = best_cluster;
  }
}

/**
 * @brief Fase de Atualização: Recalcula a posição de cada centroide como a
 * média (usando divisão inteira) de todos os pontos atribuídos ao seu cluster.
 */
void update_centroids(Point *points, Point *centroids, int M, int K, int D) {
  long long *cluster_sums = (long long *)calloc(K * D, sizeof(long long));
  int *cluster_counts = (int *)calloc(K, sizeof(int));

  for (int i = 0; i < M; i++) {
    int cluster_id = points[i].cluster_id;
    cluster_counts[cluster_id]++;
    for (int j = 0; j < D; j++) {
      cluster_sums[cluster_id * D + j] += points[i].coords[j];
    }
  }

  for (int i = 0; i < K; i++) {
    if (cluster_counts[i] > 0) {
      for (int j = 0; j < D; j++) {
        // Divisão inteira para manter os centroides em coordenadas discretas
        centroids[i].coords[j] = cluster_sums[i * D + j] / cluster_counts[i];
      }
    }
  }

  free(cluster_sums);
  free(cluster_counts);
}

/**
 * @brief Imprime os resultados finais e o checksum (como long long).
 */
void print_results(Point *centroids, int K, int D) {
  printf("--- Centroides Finais ---\n");
  long long checksum = 0;
  for (int i = 0; i < K; i++) {
    printf("Centroide %d: [", i);
    for (int j = 0; j < D; j++) {
      printf("%d", centroids[i].coords[j]);
      if (j < D - 1)
        printf(", ");
      checksum += centroids[i].coords[j];
    }
    printf("]\n");
  }
  printf("\n--- Checksum ---\n");
  printf("%lld\n", checksum); // %lld para long long int
}

/**
 * @brief Calcula e imprime o tempo de execução e o checksum final.
 * A saída é formatada para ser facilmente lida por scripts:
 * Linha 1: Tempo de execução em segundos (double)
 * Linha 2: Checksum final (long long)
 */
void print_time_and_checksum(Point *centroids, int K, int D, double exec_time) {
  long long checksum = 0;
  for (int i = 0; i < K; i++) {
    for (int j = 0; j < D; j++) {
      checksum += centroids[i].coords[j];
    }
  }
  // Saída formatada para o avaliador
  printf("%lf\n", exec_time);
  printf("%lld\n", checksum);
}

// --- Função Principal ---

int main(int argc, char *argv[]) {
  // Validação e leitura dos argumentos de linha de comando
  if (argc != 6) {
    fprintf(stderr,
            "Uso: %s <arquivo_dados> <M_pontos> <D_dimensoes> <K_clusters> "
            "<I_iteracoes>\n",
            argv[0]);
    return EXIT_FAILURE;
  }

  const char *filename = argv[1]; // Nome do arquivo de dados
  const int M = atoi(argv[2]);    // Número de pontos
  const int D = atoi(argv[3]);    // Número de dimensões
  const int K = atoi(argv[4]);    // Número de clusters
  const int I = atoi(argv[5]);    // Número de iterações

  if (M <= 0 || D <= 0 || K <= 0 || I <= 0 || K > M) {
    fprintf(stderr,
            "Erro nos parâmetros. Verifique se M,D,K,I > 0 e K <= M.\n");
    return EXIT_FAILURE;
  }

  int comm_sz;
  int my_rank;
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  struct timespec start, end;

  int points_per_proc = M / comm_sz;

  // --- Alocação de Memória ---
  int *all_coords = NULL;
  Point *points = NULL;
  Point *centroids = (Point *)malloc(K * sizeof(Point));
  int *centroid_coords = (int *)malloc(K * D * sizeof(int));

  // Alocação local para cada processo
  int *my_coords = (int *)malloc(points_per_proc * D * sizeof(int));
  Point *my_points = (Point *)malloc(points_per_proc * sizeof(Point));

  /* Buffers para redução por cluster (somatórios e contagens) */
  long long *local_sums = (long long *)calloc(K * D, sizeof(long long));
  long long *global_sums = (long long *)malloc(K * D * sizeof(long long));
  int *local_counts = (int *)calloc(K, sizeof(int));
  int *global_counts = (int *)malloc(K * sizeof(int));

  if (centroids == NULL || centroid_coords == NULL || my_coords == NULL ||
      my_points == NULL || local_sums == NULL || global_sums == NULL ||
      local_counts == NULL || global_counts == NULL) {
    fprintf(stderr, "Erro na alocação de memória.\n");
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  for (int i = 0; i < points_per_proc; i++) {
    my_points[i].coords = &my_coords[i * D];
  }
  for (int i = 0; i < K; i++) {
    centroids[i].coords = &centroid_coords[i * D];
  }

  if (my_rank == 0) {
    all_coords = (int *)malloc(M * D * sizeof(int));
    points = (Point *)malloc(M * sizeof(Point));
    if (points == NULL || all_coords == NULL) {
      fprintf(stderr, "Erro na alocação de memória.\n");
      MPI_Finalize();
      return EXIT_FAILURE;
    }
    for (int i = 0; i < M; i++) {
      points[i].coords = &all_coords[i * D];
    }
  }

  if (my_rank == 0) {
    // --- Preparação (Fora da medição de tempo) ---
    read_data_from_file(filename, points, M, D);
    initialize_centroids(points, centroids, M, K, D);

    // --- Medição de Tempo do Algoritmo Principal ---
    clock_gettime(CLOCK_MONOTONIC, &start); // Inicia o cronômetro
  }
  // Laço principal do K-Means (A única parte que será medida)
  for (int iter = 0; iter < I; iter++) {
    // Broadcast dos centroides para todos os processos
    MPI_Bcast(centroid_coords, K * D, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter dos pontos para todos os processos
    MPI_Scatter(all_coords, points_per_proc * D, MPI_INT, my_coords,
                points_per_proc * D, MPI_INT, 0, MPI_COMM_WORLD);

    // Zera os buffers locais
    memset(local_sums, 0, K * D * sizeof(long long));
    memset(local_counts, 0, K * sizeof(int));

    // Cada processo calcula as atribuições para seus pontos
    assign_points_to_clusters(my_points, centroids, points_per_proc, K, D);

    // Acumula somatórios e contagens locais por cluster
    for (int i = 0; i < points_per_proc; i++) {
      int cid = my_points[i].cluster_id;
      local_counts[cid]++;
      for (int j = 0; j < D; j++) {
        local_sums[cid * D + j] += my_points[i].coords[j];
      }
    }

    // Reduz (soma) as somas e contagens em todos os processos
    MPI_Allreduce(local_sums, global_sums, K * D, MPI_LONG_LONG, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(local_counts, global_counts, K, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);

    // Atualiza centroides a partir dos resultados reduzidos (todos os procs)
    for (int c = 0; c < K; c++) {
      if (global_counts[c] > 0) {
        for (int j = 0; j < D; j++) {
          centroid_coords[c * D + j] =
              (int)(global_sums[c * D + j] / global_counts[c]);
        }
      }
      /* Se global_counts[c] == 0, mantém o centroide anterior */
    }
  }

  if (my_rank == 0) {
    clock_gettime(CLOCK_MONOTONIC, &end); // Para o cronômetro

    // Calcula o tempo decorrido em segundos
    double time_taken =
        (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);

    // --- Apresentação dos Resultados ---
    // print_results(centroids, K, D);
    print_time_and_checksum(centroids, K, D, time_taken);

    // --- Limpeza ---
    free(all_coords);
    free(points);
  }

  // Limpeza comum a todos os processos
  free(centroid_coords);
  free(centroids);
  free(my_coords);
  free(my_points);
  free(local_sums);
  free(global_sums);
  free(local_counts);
  free(global_counts);

  //   MPI_Type_free(&MPI_Point);
  MPI_Finalize();
  return EXIT_SUCCESS;
}