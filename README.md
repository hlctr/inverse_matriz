# inverse_matriz
Inversão de Matrizes com Gauss-Jordan (Serial e Paralelo)
Este projeto implementa a inversão de matrizes quadradas utilizando o método de Gauss-Jordan com duas abordagens: serial e paralela (OpenMP). 
O objetivo é comparar o desempenho das abordagens com diferentes tamanhos de matrizes e número de threads, contribuindo para estudos de Computação de Alto Desempenho.

Estrutura do Projeto
01_Serial: Código fonte serial com orientação a linhas e colunas.
02_Parallel_openmp: Código paralelo utilizando OpenMP.
Complementos: Arquivos auxiliares e resultados.
README: Documentação principal.

Compilação
Serial
gcc -o im_serial im_serial.c -lm

Paralelo (OpenMP)
gcc -o im_parallel im_parallel.c -fopenmp -lm

Execução
Serial
./im_serial <tamanho_da_matriz> <orientacao>
<tamanho_da_matriz>: Tamanho N da matriz NxN.

<orientacao>:

1 = orientação por linhas
2 = orientação por colunas

Paralelo
./im_parallel <tamanho_da_matriz> <num_threads>
Saída
Criação ou leitura de arquivo binário com matriz original

Cálculo e validação da inversa

Armazenamento da inversa e dos tempos em arquivos .bin e .csv

Validação
Após a inversão, é realizada uma multiplicação da matriz original pela sua inversa.
O resultado é comparado com a matriz identidade para validar a operação.

Análise de Desempenho
Os arquivos .csv gerados podem ser usados para plotar gráficos de tempo de execução e speedup, 
permitindo avaliar a escalabilidade das abordagens implementadas.
