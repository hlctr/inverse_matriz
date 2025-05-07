# Implementação OpenCL para Inversão de Matriz usando Gauss-Jordan

## Visão Geral

Este projeto implementa o algoritmo de Gauss-Jordan para inversão de matrizes utilizando três abordagens:

1. **Implementação Serial** (im_serial.c)
2. **Implementação Paralela com OpenMP** (im_parallel.c)
3. **Implementação Paralela com OpenCL** (im_opencl.c)

O objetivo é demonstrar e comparar o desempenho das diferentes abordagens de programação paralela em hardware heterogêneo (CPU/GPU).

## Requisitos

- GCC (Compilador C)
- OpenMP
- OpenCL
- Bibliotecas auxiliares do OpenCL (err_code.h, wtime.c)

## Estrutura do Código

A implementação OpenCL (im_opencl.c) é uma evolução das versões serial e OpenMP, adaptada para execução em GPU. O algoritmo de Gauss-Jordan foi decomposto em kernels que podem ser executados em paralelo na GPU:

- **normalize_pivot_row**: Normaliza a linha do pivô dividindo todos os elementos pelo valor do pivô
- **elimination**: Realiza a eliminação gaussiana nas demais linhas da matriz

## Estratégia de Paralelização

A implementação OpenCL adota uma abordagem híbrida, dividindo as tarefas entre CPU e GPU:

1. **CPU**: Operações sequenciais como seleção do pivô e troca de linhas
2. **GPU**: Operações paralelizáveis como normalização da linha do pivô e eliminação gaussiana

Esta divisão foi escolhida porque:

- A seleção do pivô envolve operações de redução que são menos eficientes em GPU
- As operações de troca de linhas têm dependência sequencial
- A normalização e eliminação gaussiana têm paralelismo de dados adequado para GPU

## Otimizações Implementadas

1. **Minimização de transferências de dados**: Apenas as transferências essenciais entre host e device são realizadas
2. **Pivotamento parcial**: Implementação com seleção do maior pivô para estabilidade numérica
3. **Balanceamento de carga**: Divisão apropriada entre operações na CPU e na GPU
4. **Gerenciamento de memória eficiente**: Alocação e liberação apropriada dos recursos OpenCL
5. **Detecção automática de dispositivos**: Priorização de GPU, com fallback para CPU quando necessário

## Estrutura de Arquivos

- `im_opencl.c`: Implementação principal usando OpenCL
- `err_code.h`: Rotinas para manipulação de erros OpenCL
- `wtime.c`: Função para medição de tempo
- `results_opencl.csv`: Arquivo de resultados para análise de desempenho

## Compilação e Execução

```bash
# Compilação
gcc -o im_opencl im_opencl.c wtime.c -lOpenCL -lm

# Execução (onde N é o tamanho da matriz)
./im_opencl N
```

## Validação

O programa verifica automaticamente se a matriz inversa calculada é válida multiplicando A × A⁻¹ e verificando se o resultado é aproximadamente igual à matriz identidade. Uma tolerância de 1e-6 é usada para acomodar erros de ponto flutuante.

## Coleta e Análise de Resultados

O programa gera automaticamente um arquivo CSV (`results_opencl.csv`) que registra o tamanho da matriz e o tempo de execução correspondente. Isso permite análises de escalabilidade para diferentes dimensões de matriz.

## Considerações de Desempenho

- Para matrizes pequenas (n < 1000), a sobrecarga de inicialização do OpenCL pode superar os ganhos de paralelização
- Para matrizes grandes (n ≥ 1000), a implementação OpenCL oferece ganhos significativos de desempenho em relação às implementações serial e OpenMP
- O algoritmo é especialmente eficaz em GPUs com muitos núcleos de processamento

## Limitações Conhecidas

1. Matrizes muito grandes podem exceder a memória disponível na GPU
2. O desempenho pode variar significativamente dependendo do hardware específico utilizado
3. Matrizes singulares ou mal condicionadas podem levar a resultados imprecisos
