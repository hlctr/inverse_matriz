#!/bin/bash

# Compile o programa
gcc -O3 -fopenmp -o inversion_omp im_parallel.c -lm

# Tamanhos de matriz para testar
SIZES=(10 100 500 1000 2000 3000 4000)

# Número de threads para testar
THREADS=(1 2 4 8 16)

# Limpa o arquivo de resultados se existir
> results_omp.csv

echo "Iniciando benchmarks..."

# Loop pelos diferentes tamanhos de matriz
for n in "${SIZES[@]}"; do
    echo "Testando matriz de tamanho $n x $n"
    
    # Loop pelo número de threads
    for t in "${THREADS[@]}"; do
        echo "  Com $t threads..."
        
        # Executa o programa 3 vezes e calcula a média
        total_time=0
        runs=3
        
        for (( i=1; i<=$runs; i++ )); do
            echo "    Execução $i de $runs"
            ./inversion_omp $n $t
        done
    done
done

echo "Benchmarks concluídos. Resultados salvos em results_omp.csv"