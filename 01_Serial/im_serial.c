#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>

// Medir o tempo em segundos
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Gerar uma matriz aleatória n x n que seja inversível
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

// Salva a matriz em um arquivo .bin
void save_matrix_to_file(double *matrix, int n, const char *filename) {
    FILE *file = fopen(filename, "wb");
    if (file == NULL) {
        fprintf(stderr, "Erro ao abrir o arquivo %s para escrita\n", filename);
        exit(EXIT_FAILURE);
    }
    
    fwrite(matrix, sizeof(double), n*n, file);
    fclose(file);
}

// Ler a matriz em um arquivo .bin salva anteriormente
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

// Função para imprimir matriz (apenas para depuração)
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

// Função para calcular a inversa da matriz usando o método de Gauss-Jordan
// Orientação a linhas
void calculate_inverse_row_oriented(double *A, double *Ainv, int n) {
    // Cria uma cópia da matriz A para não modificá-la
    double *temp_A = (double*)malloc(n*n*sizeof(double));
    memcpy(temp_A, A, n*n*sizeof(double));
    
    // Inicializa Ainv como matriz identidade
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
        
        for (int i = k + 1; i < n; i++) {
            double abs_value = fabs(temp_A[i*n + k]);
            if (abs_value > pivot_value) {
                pivot_value = abs_value;
                pivot_row = i;
            }
        }
        
        // Se o pivô for muito pequeno, a matriz pode ser singular
        if (pivot_value < 1e-10) {
            fprintf(stderr, "Erro: A matriz parece ser singular ou mal condicionada\n");
            free(temp_A);
            exit(EXIT_FAILURE);
        }
        
        // Troca as linhas se necessário
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
        
        // Normaliza a linha do pivô
        double pivot = temp_A[k*n + k];
        for (int j = 0; j < n; j++) {
            temp_A[k*n + j] /= pivot;
            Ainv[k*n + j] /= pivot;
        }
        
        // Eliminação de Gauss
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

// Função para calcular a inversa da matriz usando o método de Gauss-Jordan
// Orientação a colunas
void calculate_inverse_column_oriented(double *A, double *Ainv, int n) {
    // Cria uma cópia da matriz A para não modificá-la
    double *temp_A = (double*)malloc(n*n*sizeof(double));
    memcpy(temp_A, A, n*n*sizeof(double));
    
    // Inicializa Ainv como matriz identidade
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Ainv[i*n + j] = (i == j) ? 1.0 : 0.0;
        }
    }
    
    // Algoritmo de Gauss-Jordan (orientado a colunas)
    for (int k = 0; k < n; k++) {
        // Encontra o pivô (valor máximo na coluna k)
        int pivot_row = k;
        double pivot_value = fabs(temp_A[k*n + k]);
        
        for (int i = k + 1; i < n; i++) {
            double abs_value = fabs(temp_A[i*n + k]);
            if (abs_value > pivot_value) {
                pivot_value = abs_value;
                pivot_row = i;
            }
        }
        
        // Se o pivô for muito pequeno, a matriz pode ser singular
        if (pivot_value < 1e-10) {
            fprintf(stderr, "Erro: A matriz parece ser singular ou mal condicionada\n");
            free(temp_A);
            exit(EXIT_FAILURE);
        }
        
        // Troca as linhas se necessário
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
        
        // Normaliza a linha do pivô
        double pivot = temp_A[k*n + k];
        for (int j = 0; j < n; j++) {
            temp_A[k*n + j] /= pivot;
            Ainv[k*n + j] /= pivot;
        }
        
        // Loop externo sobre colunas (diferente da orientação a linhas)
        for (int j = 0; j < n; j++) {
            if (j != k) {
                for (int i = 0; i < n; i++) {
                    if (i != k) {
                        temp_A[i*n + j] -= temp_A[i*n + k] * temp_A[k*n + j];
                        Ainv[i*n + j] -= Ainv[i*n + k] * temp_A[k*n + j];
                    }
                }
            }
        }
    }
    
    free(temp_A);
}

// Valida a inversa calculada (A * A^-1 deve ser aproximadamente I)
int validate_inverse(double *A, double *Ainv, int n) {
    double *result = (double*)malloc(n*n*sizeof(double));
    double epsilon = 1e-6;
    
    // Calcula A * A^-1
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
        fprintf(stderr, "Uso: %s <tamanho_da_matriz> <orientacao>\n", argv[0]);
        fprintf(stderr, "orientacao: 1 para orientado a linhas, 2 para orientado a colunas\n");
        return EXIT_FAILURE;
    }
    
    // Obtem o tamanho da matriz e orientação dos argumentos da linha de comando
    int n = atoi(argv[1]);
    int orientation = atoi(argv[2]);
    
    if (n <= 0) {
        fprintf(stderr, "Erro: O tamanho da matriz deve ser positivo\n");
        return EXIT_FAILURE;
    }
    
    if (orientation != 1 && orientation != 2) {
        fprintf(stderr, "Erro: Orientação deve ser 1 (linhas) ou 2 (colunas)\n");
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
    sprintf(output_filename, "inverse_matrix_%d_%s.bin", n, orientation == 1 ? "row" : "col");
    
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
    
    // Calcula a matriz inversa com base na orientação escolhida
    if (orientation == 1) {
        printf("Calculando inversa (orientação a linhas)...\n");
        calculate_inverse_row_oriented(A, Ainv, n);
    } else {
        printf("Calculando inversa (orientação a colunas)...\n");
        calculate_inverse_column_oriented(A, Ainv, n);
    }
    
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
    sprintf(results_filename, "results_%s.csv", orientation == 1 ? "row" : "col");
    
    FILE *results_file = fopen(results_filename, "a");
    if (results_file == NULL) {
        fprintf(stderr, "Erro ao abrir o arquivo de resultados %s\n", results_filename);
    } else {
        // Verifica se o arquivo está vazio para adicionar o cabeçalho
        fseek(results_file, 0, SEEK_END);
        long size = ftell(results_file);
        
        if (size == 0) {
            fprintf(results_file, "tamanho_matriz,tempo_execucao\n");
        }
        
        // Adiciona os resultados
        fprintf(results_file, "%d,%.6f\n", n, execution_time);
        fclose(results_file);
    }
    
    printf("Tamanho da matriz: %d x %d\n", n, n);
    printf("Tempo de execução: %.6f segundos\n", execution_time);
    
    // Libera a memória
    free(A);
    free(Ainv);
    
    return EXIT_SUCCESS;
}