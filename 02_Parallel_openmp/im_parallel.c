#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <omp.h>  // Inclusão da biblioteca OpenMP

// Função para medir o tempo em segundos
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Função para gerar uma matriz aleatória n x n que seja inversível
void generate_invertible_matrix(double *matrix, int n) {
    // Primeiro cria uma matriz diagonal com valores não nulos na diagonal
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                matrix[i*n + j] = (double)(rand() % 100) + 1.0; // Valores de 1 a 100 na diagonal
            } else {
                matrix[i*n + j] = 0.0;
            }
        }
    }
    
    // Aplica permutações aleatórias para manter a matriz inversível mas não trivial
    for (int k = 0; k < n*2; k++) {
        int row1 = rand() % n;
        int row2 = rand() % n;
        
        if (row1 != row2) {
            // Soma a linha row1 com a linha row2 multiplicada por um fator aleatório
            double factor = (double)(rand() % 10) + 0.1;
            for (int j = 0; j < n; j++) {
                matrix[row1*n + j] += factor * matrix[row2*n + j];
            }
        }
    }
}

// Função para salvar a matriz em um arquivo binário
void save_matrix_to_file(double *matrix, int n, const char *filename) {
    FILE *file = fopen(filename, "wb");
    if (file == NULL) {
        fprintf(stderr, "Erro ao abrir o arquivo %s para escrita\n", filename);
        exit(EXIT_FAILURE);
    }
    
    fwrite(matrix, sizeof(double), n*n, file);
    fclose(file);
}

// Função para carregar a matriz de um arquivo binário
void load_matrix_from_file(double *matrix, int n, const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Erro ao abrir o arquivo %s para leitura\n", filename);
        exit(EXIT_FAILURE);
    }
    
    size_t read_elements = fread(matrix, sizeof(double), n*n, file);
    if (read_elements != n*n) {
        fprintf(stderr, "Erro ao ler dados do arquivo %s\n", filename);
        exit(EXIT_FAILURE);
    }
    
    fclose(file);
}

// Função para imprimir matriz (para depuração)
void print_matrix(double *matrix, int n, const char *label) {
    printf("%s:\n", label);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%10.4f ", matrix[i*n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Função paralela para calcular a inversa da matriz usando o método de Gauss-Jordan
// Orientação a linhas (row-oriented)
void calculate_inverse_row_oriented_parallel(double *A, double *Ainv, int n, int num_threads) {
    // Define o número de threads a ser usado
    omp_set_num_threads(num_threads);
    
    // Cria uma cópia da matriz A para não modificá-la
    double *temp_A = (double*)malloc(n*n*sizeof(double));
    memcpy(temp_A, A, n*n*sizeof(double));
    
    // Inicializa Ainv como matriz identidade (paralelizado)
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Ainv[i*n + j] = (i == j) ? 1.0 : 0.0;
        }
    }
    
    // Algoritmo de Gauss-Jordan
    for (int k = 0; k < n; k++) {
        // Encontra o pivô (valor máximo na coluna k)
        int pivot_row = k;
        double pivot_value = fabs(temp_A[k*n + k]);
        
        // Usando redução para encontrar o pivô em paralelo
        #pragma omp parallel
        {
            int local_pivot_row = pivot_row;
            double local_pivot_value = pivot_value;
            
            #pragma omp for nowait
            for (int i = k + 1; i < n; i++) {
                double abs_value = fabs(temp_A[i*n + k]);
                if (abs_value > local_pivot_value) {
                    local_pivot_value = abs_value;
                    local_pivot_row = i;
                }
            }
            
            // Redução crítica para encontrar o pivô global
            #pragma omp critical
            {
                if (local_pivot_value > pivot_value) {
                    pivot_value = local_pivot_value;
                    pivot_row = local_pivot_row;
                }
            }
        }
        
        // Se o pivô for muito pequeno, a matriz pode ser singular
        if (pivot_value < 1e-10) {
            fprintf(stderr, "Erro: A matriz parece ser singular ou mal condicionada\n");
            free(temp_A);
            exit(EXIT_FAILURE);
        }
        
        // Troca as linhas se necessário (serial, pois é dependente)
        if (pivot_row != k) {
            for (int j = 0; j < n; j++) {
                double temp = temp_A[k*n + j];
                temp_A[k*n + j] = temp_A[pivot_row*n + j];
                temp_A[pivot_row*n + j] = temp;
                
                temp = Ainv[k*n + j];
                Ainv[k*n + j] = Ainv[pivot_row*n + j];
                Ainv[pivot_row*n + j] = temp;
            }
        }
        
        // Normaliza a linha do pivô (paralelizado)
        double pivot = temp_A[k*n + k];
        #pragma omp parallel for
        for (int j = 0; j < n; j++) {
            temp_A[k*n + j] /= pivot;
            Ainv[k*n + j] /= pivot;
        }
        
        // Eliminação de Gauss (paralelizado)
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n; i++) {
            if (i != k) {
                double factor = temp_A[i*n + k];
                for (int j = 0; j < n; j++) {
                    temp_A[i*n + j] -= factor * temp_A[k*n + j];
                    Ainv[i*n + j] -= factor * Ainv[k*n + j];
                }
            }
        }
    }
    
    free(temp_A);
}

// Função para validar a inversa calculada (A * A^-1 deve ser aproximadamente I)
int validate_inverse(double *A, double *Ainv, int n) {
    double *result = (double*)malloc(n*n*sizeof(double));
    double epsilon = 1e-6;
    
    // Calcula A * A^-1 (paralelizado)
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i*n + j] = 0.0;
            for (int k = 0; k < n; k++) {
                result[i*n + j] += A[i*n + k] * Ainv[k*n + j];
            }
        }
    }
    
    // Verifica se o resultado é aproximadamente a matriz identidade
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double expected = (i == j) ? 1.0 : 0.0;
            if (fabs(result[i*n + j] - expected) > epsilon) {
                free(result);
                return 0; // Falha na validação
            }
        }
    }
    
    free(result);
    return 1; // Validação bem-sucedida
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Uso: %s <tamanho_da_matriz> <num_threads>\n", argv[0]);
        return EXIT_FAILURE;
    }
    
    // Obtem o tamanho da matriz e número de threads dos argumentos
    int n = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    
    if (n <= 0) {
        fprintf(stderr, "Erro: O tamanho da matriz deve ser positivo\n");
        return EXIT_FAILURE;
    }
    
    if (num_threads <= 0) {
        fprintf(stderr, "Erro: O número de threads deve ser positivo\n");
        return EXIT_FAILURE;
    }
    
    // Aloca memória para as matrizes
    double *A = (double*)malloc(n*n*sizeof(double));
    double *Ainv = (double*)malloc(n*n*sizeof(double));
    
    if (A == NULL || Ainv == NULL) {
        fprintf(stderr, "Erro: Falha na alocação de memória\n");
        free(A);
        free(Ainv);
        return EXIT_FAILURE;
    }
    
    // Define o nome dos arquivos de entrada e saída
    char input_filename[100], output_filename[100];
    sprintf(input_filename, "matrix_%d.bin", n);
    sprintf(output_filename, "inverse_matrix_%d_omp_%d.bin", n, num_threads);
    
    // Verifica se o arquivo de entrada existe, se não, gera e salva uma matriz
    FILE *test_file = fopen(input_filename, "rb");
    if (test_file == NULL) {
        printf("Arquivo de matriz de entrada não encontrado. Gerando nova matriz %dx%d...\n", n, n);
        
        // Configura a semente para números aleatórios
        srand(time(NULL));
        
        // Gera uma matriz inversível aleatória
        generate_invertible_matrix(A, n);
        
        // Salva a matriz no arquivo
        save_matrix_to_file(A, n, input_filename);
        printf("Matriz salva em %s\n", input_filename);
    } else {
        fclose(test_file);
        printf("Carregando matriz %dx%d do arquivo %s\n", n, n, input_filename);
        load_matrix_from_file(A, n, input_filename);
    }
    
    // Mede o tempo de execução
    double start_time = get_time();
    
    // Calcula a matriz inversa usando o método paralelo
    printf("Calculando inversa (paralela com OpenMP, %d threads)...\n", num_threads);
    calculate_inverse_row_oriented_parallel(A, Ainv, n, num_threads);
    
    double end_time = get_time();
    double execution_time = end_time - start_time;
    
    // Valida a matriz inversa calculada
    if (validate_inverse(A, Ainv, n)) {
        printf("Validação da matriz inversa: SUCESSO\n");
    } else {
        printf("Validação da matriz inversa: FALHA\n");
    }
    
    // Salva a matriz inversa em um arquivo
    save_matrix_to_file(Ainv, n, output_filename);
    printf("Matriz inversa salva em %s\n", output_filename);
    
    // Grava os resultados em um arquivo CSV para análise de escalabilidade
    char results_filename[100];
    sprintf(results_filename, "results_omp.csv");
    
    FILE *results_file = fopen(results_filename, "a");
    if (results_file == NULL) {
        fprintf(stderr, "Erro ao abrir o arquivo de resultados %s\n", results_filename);
    } else {
        // Verifica se o arquivo está vazio para adicionar o cabeçalho
        fseek(results_file, 0, SEEK_END);
        long size = ftell(results_file);
        
        if (size == 0) {
            fprintf(results_file, "tamanho_matriz,num_threads,tempo_execucao\n");
        }
        
        // Adiciona os resultados
        fprintf(results_file, "%d,%d,%.6f\n", n, num_threads, execution_time);
        fclose(results_file);
    }
    
    printf("Tamanho da matriz: %d x %d\n", n, n);
    printf("Número de threads: %d\n", num_threads);
    printf("Tempo de execução: %.6f segundos\n", execution_time);
    
    // Libera a memória
    free(A);
    free(Ainv);
    
    return EXIT_SUCCESS;
}