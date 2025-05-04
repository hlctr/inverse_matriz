
# 🧮 Inversão de Matrizes com o Algoritmo Gauss-Jordan (Serial e Paralelo)

Este projeto implementa o algoritmo de **inversão de matrizes quadradas** utilizando o método de **Gauss-Jordan**, com três variações:

- ✅ **Serial orientado a linhas**
- ✅ **Serial orientado a colunas**
- ✅ **Paralelo com OpenMP**

O objetivo principal é **avaliar o desempenho** entre versões sequenciais e paralelas em diferentes tamanhos de matrizes e quantidades de threads, contribuindo para estudos e aplicações em **Computação de Alto Desempenho**.

## 📁 Estrutura do Projeto

```
📦 inverse_matriz/
├── im_serial.c             # Versão serial (linhas e colunas)
├── im_parallel.c           # Versão paralela com OpenMP
├── README.md               # Documentação do projeto
├── *.bin                   # Matrizes originais e invertidas
└── *.csv                   # Resultados de tempo de execução
```

## ⚙️ Compilação

### 🔹 Versão Serial
```bash
gcc -o im_serial im_serial.c -lm
```

### 🔹 Versão Paralela (OpenMP)
```bash
gcc -o im_parallel im_parallel.c -fopenmp -lm
```

## ▶️ Execução

### 🔸 Serial
```bash
./im_serial <tamanho_da_matriz> <orientacao>
```

- `<tamanho_da_matriz>`: Número inteiro positivo (ex: 500)
- `<orientacao>`:
  - `1` = orientação a linhas
  - `2` = orientação a colunas

### 🔸 Paralelo (OpenMP)
```bash
./im_parallel <tamanho_da_matriz> <num_threads>
```

- `<num_threads>`: Número de threads OpenMP (ex: 4)

## 📤 Saídas Geradas

- Arquivo `.bin` com a matriz original (ex: `matrix_500.bin`)
- Arquivo `.bin` com a matriz inversa (ex: `inverse_matrix_500_row.bin`, `inverse_matrix_500_omp_4.bin`)
- Arquivo `.csv` com resultados de tempo de execução:
  - `results_row.csv`, `results_col.csv` (serial)
  - `results_omp.csv` (paralelo)

## ✔️ Validação

Após o cálculo da inversa, é realizada a multiplicação da matriz original por sua inversa. O resultado é comparado com a **matriz identidade**, utilizando uma tolerância numérica (`epsilon = 1e-6`) para validar a correção da inversão.

## 📊 Análise de Desempenho

Os arquivos `.csv` permitem visualizar:
- Tempo de execução por variação e tamanho de matriz
- Speedup obtido com diferentes quantidades de threads

## 🧠 Conclusão

Este projeto é ideal para entender:
- Diferenças práticas entre abordagens sequenciais e paralelas
- Variações de orientação (linha x coluna) no método de Gauss-Jordan
- Impacto da paralelização com OpenMP no desempenho de operações matriciais
