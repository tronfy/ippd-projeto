#define _POSIX_C_SOURCE 199309L // Necessário para CLOCK_MONOTONIC
#include <immintrin.h>
#include <limits.h>             // Para LLONG_MAX
#include <pthread.h>            // Para pthreads
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h> // Header correto para clock_gettime e struct timespec

// Estrutura para representar um ponto no espaço D-dimensional
typedef struct {
  int *coords;    // Vetor de coordenadas inteiras
  int cluster_id; // ID do cluster ao qual o ponto pertence
} Point;

// Estrutura para passar dados para as threads
typedef struct {
  Point *points;
  Point *centroids;
  int start_idx;
  int end_idx;
  int K;
  int D;
} ThreadData;

// --- Funções Utilitárias ---

/**
 * @brief Calcula a distância Euclidiana ao quadrado entre dois pontos com
 * coordenadas inteiras. Usa 'long long' para evitar overflow no cálculo da
 * distância e da diferença.
 * @return A distância Euclidiana ao quadrado como um long long.
 */
long long euclidean_dist_sq(Point *p1, Point *p2, int D) {
  long long dist = 0;

  // #if defined(__AVX2__)
  __m256i sum_vec = _mm256_setzero_si256();
  int i = 0;

  for (; i + 7 < D; i += 8) {
    // carrega 8 valores int32 de cada ponto
    __m256i v1 = _mm256_loadu_si256((__m256i *)&p1->coords[i]);
    __m256i v2 = _mm256_loadu_si256((__m256i *)&p2->coords[i]);

    // calcula a diferença
    __m256i diff = _mm256_sub_epi32(v1, v2);

    // eleva diferença ao quadrado: converte para 64 bits para multiplicação
    __m128i diff_lo = _mm256_castsi256_si128(diff);
    __m128i diff_hi = _mm256_extracti128_si256(diff, 1);

    __m256i diff_lo_64 = _mm256_cvtepi32_epi64(diff_lo);
    __m256i diff_hi_64 = _mm256_cvtepi32_epi64(diff_hi);

    __m256i sq_lo = _mm256_mul_epi32(diff_lo_64, diff_lo_64);
    __m256i sq_hi = _mm256_mul_epi32(diff_hi_64, diff_hi_64);

    // acumula
    sum_vec = _mm256_add_epi64(sum_vec, sq_lo);
    sum_vec = _mm256_add_epi64(sum_vec, sq_hi);
  }

  // soma os elementos do vetor de soma
  long long temp[4];
  _mm256_storeu_si256((__m256i *)temp, sum_vec);
  dist = temp[0] + temp[1] + temp[2] + temp[3];

  // processa os elementos restantes
  for (; i < D; i++) {
    long long diff = (long long)p1->coords[i] - p2->coords[i];
    dist += diff * diff;
  }
  // #else
  //   for (int i = 0; i < D; i++) {
  //     long long diff = (long long)p1->coords[i] - p2->coords[i];
  //     dist += diff * diff;
  //   }
  // #endif

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
 * @brief Worker thread para atribuição de pontos aos clusters.
 */
void *assign_points_worker(void *arg) {
  ThreadData *data = (ThreadData *)arg;
  
  for (int i = data->start_idx; i < data->end_idx; i++) {
    long long min_dist = LLONG_MAX;
    int best_cluster = -1;

    for (int j = 0; j < data->K; j++) {
      long long dist = euclidean_dist_sq(&data->points[i], 
                                          &data->centroids[j], 
                                          data->D);
      if (dist < min_dist) {
        min_dist = dist;
        best_cluster = j;
      }
    }
    data->points[i].cluster_id = best_cluster;
  }
  
  return NULL;
}

/**
 * @brief Fase de Atribuição Paralela: Associa cada ponto ao cluster do centroide mais
 * próximo usando pthreads.
 */
void assign_points_to_clusters(Point *points, Point *centroids, int M, int K,
                               int D, int num_threads) {
  pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
  ThreadData *thread_data = malloc(num_threads * sizeof(ThreadData));
  
  int points_per_thread = M / num_threads;
  
  for (int i = 0; i < num_threads; i++) {
    thread_data[i].points = points;
    thread_data[i].centroids = centroids;
    thread_data[i].start_idx = i * points_per_thread;
    thread_data[i].end_idx = (i == num_threads - 1) ? M : (i + 1) * points_per_thread;
    thread_data[i].K = K;
    thread_data[i].D = D;
    
    pthread_create(&threads[i], NULL, assign_points_worker, &thread_data[i]);
  }
  
  for (int i = 0; i < num_threads; i++) {
    pthread_join(threads[i], NULL);
  }
  
  free(threads);
  free(thread_data);
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
  if (argc != 6 && argc != 7) {
    fprintf(stderr,
            "Uso: %s <arquivo_dados> <M_pontos> <D_dimensoes> <K_clusters> "
            "<I_iteracoes> [num_threads]\n",
            argv[0]);
    return EXIT_FAILURE;
  }

  const char *filename = argv[1]; // Nome do arquivo de dados
  const int M = atoi(argv[2]);    // Número de pontos
  const int D = atoi(argv[3]);    // Número de dimensões
  const int K = atoi(argv[4]);    // Número de clusters
  const int I = atoi(argv[5]);    // Número de iterações
  
  // Número de threads: usa argumento se fornecido, senão OMP_NUM_THREADS, senão padrão 4
  int num_threads = 4; // Padrão
  if (argc == 7) {
    num_threads = atoi(argv[6]);
  } else {
    char *omp_threads = getenv("OMP_NUM_THREADS");
    if (omp_threads != NULL) {
      num_threads = atoi(omp_threads);
    }
  }
  
  if (num_threads <= 0) {
    num_threads = 4; // Fallback para valor seguro
  }

  if (M <= 0 || D <= 0 || K <= 0 || I <= 0 || K > M) {
    fprintf(stderr,
            "Erro nos parâmetros. Verifique se M,D,K,I > 0 e K <= M.\n");
    return EXIT_FAILURE;
  }

  // --- Alocação de Memória ---
  int *all_coords = (int *)malloc((M + K) * D * sizeof(int));
  Point *points = (Point *)malloc(M * sizeof(Point));
  Point *centroids = (Point *)malloc(K * sizeof(Point));
  // ... (verificação de alocação) ...
  for (int i = 0; i < M; i++) {
    points[i].coords = &all_coords[i * D];
  }
  for (int i = 0; i < K; i++) {
    centroids[i].coords = &all_coords[(M + i) * D];
  }

  // --- Preparação (Fora da medição de tempo) ---
  read_data_from_file(filename, points, M, D);
  initialize_centroids(points, centroids, M, K, D);

  // --- Medição de Tempo do Algoritmo Principal ---
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start); // Inicia o cronômetro

  // Laço principal do K-Means (A única parte que será medida)
  for (int iter = 0; iter < I; iter++) {
    assign_points_to_clusters(points, centroids, M, K, D, num_threads);
    update_centroids(points, centroids, M, K, D);
  }

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
  free(centroids);

  return EXIT_SUCCESS;
}
