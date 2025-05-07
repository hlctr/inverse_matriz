/*
 * im_opencl.c - Implementação otimizada de inversão de matriz usando Gauss-Jordan em OpenCL
 */

#define CL_TARGET_OPENCL_VERSION 120

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "err_code.h"

double wtime(void);

#define MAX_SOURCE_SIZE (0x100000)

size_t round_up(size_t value, size_t multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

const char *opencl_kernel_source = "\n" \
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n\n" \
"__kernel void init_identity(__global double* I, const int n) {\n" \
"    int i = get_global_id(0);\n" \
"    int j = get_global_id(1);\n" \
"    if (i < n && j < n) I[i * n + j] = (i == j) ? 1.0 : 0.0;\n" \
"}\n\n" \
"__kernel void find_pivot(__global double* A,\n" \
"                         const int n,\n" \
"                         const int k,\n" \
"                         __global double* pivot_vals) {\n" \
"    int i = get_global_id(0) + k;\n" \
"    if (i < n) {\n" \
"        pivot_vals[i-k] = fabs(A[i * n + k]);\n" \
"    }\n" \
"}\n\n" \
"__kernel void swap_rows(__global double* matrix, const int n, const int row1, const int row2) {\n" \
"    int j = get_global_id(0);\n" \
"    if (j < n) {\n" \
"        double temp = matrix[row1 * n + j];\n" \
"        matrix[row1 * n + j] = matrix[row2 * n + j];\n" \
"        matrix[row2 * n + j] = temp;\n" \
"    }\n" \
"}\n\n" \
"__kernel void normalize_row(__global double* A, __global double* I, const int n, const int k) {\n" \
"    int j = get_global_id(0);\n" \
"    if (j < n) {\n" \
"        double pivot = A[k * n + k];\n" \
"        if (pivot != 0.0) {\n" \
"            A[k * n + j] /= pivot;\n" \
"            I[k * n + j] /= pivot;\n" \
"        }\n" \
"    }\n" \
"}\n\n" \
"__kernel void eliminate_row(__global double* A,\n" \
"                           __global double* I,\n" \
"                           const int n,\n" \
"                           const int k) {\n" \
"    int i = get_global_id(0);\n" \
"    int j = get_global_id(1);\n" \
"    \n" \
"    if (i < n && j < n && i != k) {\n" \
"        double factor = A[i * n + k];\n" \
"        A[i * n + j] -= factor * A[k * n + j];\n" \
"        I[i * n + j] -= factor * I[k * n + j];\n" \
"    }\n" \
"}\n\n" \
"__kernel void verify_result(__global double* A_original,\n" \
"                            __global double* I,\n" \
"                            __global double* result,\n" \
"                            const int n) {\n" \
"    int i = get_global_id(0);\n" \
"    int j = get_global_id(1);\n" \
"    if (i < n && j < n) {\n" \
"        double sum = 0.0;\n" \
"        for (int k = 0; k < n; k++) sum += A_original[i * n + k] * I[k * n + j];\n" \
"        result[i * n + j] = sum;\n" \
"    }\n" \
"}\n";

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Uso: %s <tamanho_da_matriz>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    if (n <= 0) {
        printf("Erro: O tamanho da matriz deve ser maior que zero.\n");
        return 1;
    }

    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_program program = NULL;
    cl_kernel kernels[6];
    cl_mem a_mem_obj = NULL, i_mem_obj = NULL, result_mem_obj = NULL, a_original_mem_obj = NULL;
    cl_mem pivot_vals_mem = NULL;
    cl_int err;
    cl_uint num_platforms, num_devices;

    err = clGetPlatformIDs(1, &platform_id, &num_platforms);
    checkError(err, "clGetPlatformIDs");

    cl_device_type device_type = CL_DEVICE_TYPE_GPU;
    err = clGetDeviceIDs(platform_id, device_type, 1, &device_id, &num_devices);
    if (err == CL_DEVICE_NOT_FOUND) {
        printf("GPU não encontrada, tentando CPU...\n");
        device_type = CL_DEVICE_TYPE_CPU;
        err = clGetDeviceIDs(platform_id, device_type, 1, &device_id, &num_devices);
        checkError(err, "clGetDeviceIDs (CPU)");
    } else {
        checkError(err, "clGetDeviceIDs (GPU)");
    }

    char device_name[1024];
    err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    checkError(err, "clGetDeviceInfo");
    printf("Dispositivo: %s\n", device_name);

    // Obter tamanho máximo do work group
    size_t max_work_group_size;
    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);
    checkError(err, "clGetDeviceInfo (max work group)");

    cl_ulong global_mem_size;
    err = clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, NULL);
    checkError(err, "clGetDeviceInfo (memory)");
    printf("Memória global disponível: %.2f GB\n", global_mem_size / (1024.0 * 1024.0 * 1024.0));

    size_t matriz_size = 3 * n * n * sizeof(double);
    if (matriz_size > global_mem_size * 0.8) {
        printf("Aviso: O tamanho da matriz (%zu bytes) pode exceder a memória disponível.\n", matriz_size);
        if (matriz_size > global_mem_size) {
            printf("Erro: Matriz muito grande para a memória disponível. Tente reduzir o tamanho.\n");
            return 1;
        }
    }

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    checkError(err, "clCreateContext");

    #ifdef CL_VERSION_2_0
    command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
    #else
    command_queue = clCreateCommandQueue(context, device_id, 0, &err);
    #endif
    checkError(err, "clCreateCommandQueue");

    program = clCreateProgramWithSource(context, 1, (const char **)&opencl_kernel_source, NULL, &err);
    checkError(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("Erro de compilação:\n%s\n", log);
        free(log);
        return 1;
    }

    const char* kernel_names[] = {
        "init_identity", "find_pivot", "swap_rows",
        "normalize_row", "eliminate_row", "verify_result"
    };
    for (int i = 0; i < 6; i++) {
        kernels[i] = clCreateKernel(program, kernel_names[i], &err);
        checkError(err, "clCreateKernel");
    }

    double *A = (double *)malloc(n * n * sizeof(double));
    double *I = (double *)malloc(n * n * sizeof(double));
    double *result = (double *)malloc(n * n * sizeof(double));
    double *pivot_vals = (double *)malloc(n * sizeof(double));

    if (!A || !I || !result || !pivot_vals) {
        fprintf(stderr, "Erro: falha na alocação de memória no host.\n");
        exit(EXIT_FAILURE);
    }

    // Inicializar matriz A com valores não singulares
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] = (i == j) ? (double)(rand() % 100 + n + 1) : (double)(rand() % 10) * 0.1;
        }
    }

    a_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, n * n * sizeof(double), NULL, &err);
    checkError(err, "clCreateBuffer (A)");
    i_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, n * n * sizeof(double), NULL, &err);
    checkError(err, "clCreateBuffer (I)");
    result_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * n * sizeof(double), NULL, &err);
    checkError(err, "clCreateBuffer (result)");
    pivot_vals_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(double), NULL, &err);
    checkError(err, "clCreateBuffer (pivot_vals)");
    a_original_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, n * n * sizeof(double), NULL, &err);
    checkError(err, "clCreateBuffer (A_original)");

    // Escrever dados nos buffers
    err = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0, n * n * sizeof(double), A, 0, NULL, NULL);
    checkError(err, "clEnqueueWriteBuffer (A)");
    err = clEnqueueWriteBuffer(command_queue, a_original_mem_obj, CL_TRUE, 0, n * n * sizeof(double), A, 0, NULL, NULL);
    checkError(err, "clEnqueueWriteBuffer (A_original)");

    // Configurar work sizes para kernels 2D
    size_t local_work_size[2];
    local_work_size[0] = local_work_size[1] = 16;
    if (n < 16) {
        local_work_size[0] = local_work_size[1] = 1;
    }
    
    size_t global_work_size[2] = {
        round_up(n, local_work_size[0]),
        round_up(n, local_work_size[1])
    };

    // Inicializar matriz identidade
    err = clSetKernelArg(kernels[0], 0, sizeof(cl_mem), &i_mem_obj);
    err |= clSetKernelArg(kernels[0], 1, sizeof(int), &n);
    checkError(err, "clSetKernelArg (init_identity)");

    err = clEnqueueNDRangeKernel(command_queue, kernels[0], 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    checkError(err, "clEnqueueNDRangeKernel (init_identity)");
    clFinish(command_queue);

    double start_time = wtime();

    // Algoritmo principal
    for (int k = 0; k < n; k++) {
        // 1. Encontrar pivô
        err = clSetKernelArg(kernels[1], 0, sizeof(cl_mem), &a_mem_obj);
        err |= clSetKernelArg(kernels[1], 1, sizeof(int), &n);
        err |= clSetKernelArg(kernels[1], 2, sizeof(int), &k);
        err |= clSetKernelArg(kernels[1], 3, sizeof(cl_mem), &pivot_vals_mem);
        checkError(err, "clSetKernelArg (find_pivot)");

        size_t global_find = round_up(n - k, 256);
        size_t local_find = 256;
        if (global_find == 0) global_find = local_find;

        err = clEnqueueNDRangeKernel(command_queue, kernels[1], 1, NULL, &global_find, &local_find, 0, NULL, NULL);
        checkError(err, "clEnqueueNDRangeKernel (find_pivot)");
        clFinish(command_queue);

        // Ler valores de pivô
        err = clEnqueueReadBuffer(command_queue, pivot_vals_mem, CL_TRUE, 0, (n - k) * sizeof(double), pivot_vals, 0, NULL, NULL);
        checkError(err, "clEnqueueReadBuffer (pivot_vals)");

        // Encontrar máximo no host
        int max_idx = 0;
        double max_val = pivot_vals[0];
        for (int i = 1; i < (n - k); i++) {
            if (pivot_vals[i] > max_val) {
                max_val = pivot_vals[i];
                max_idx = i;
            }
        }
        max_idx += k;

        // 2. Trocar linhas se necessário
        if (max_idx != k) {
            err = clSetKernelArg(kernels[2], 0, sizeof(cl_mem), &a_mem_obj);
            err |= clSetKernelArg(kernels[2], 1, sizeof(int), &n);
            err |= clSetKernelArg(kernels[2], 2, sizeof(int), &k);
            err |= clSetKernelArg(kernels[2], 3, sizeof(int), &max_idx);
            checkError(err, "clSetKernelArg (swap_rows A)");

            size_t global_swap = round_up(n, 256);
            size_t local_swap = 256;
            err = clEnqueueNDRangeKernel(command_queue, kernels[2], 1, NULL, &global_swap, &local_swap, 0, NULL, NULL);
            checkError(err, "clEnqueueNDRangeKernel (swap_rows A)");
            clFinish(command_queue);

            err = clSetKernelArg(kernels[2], 0, sizeof(cl_mem), &i_mem_obj);
            checkError(err, "clSetKernelArg (swap_rows I)");
            err = clEnqueueNDRangeKernel(command_queue, kernels[2], 1, NULL, &global_swap, &local_swap, 0, NULL, NULL);
            checkError(err, "clEnqueueNDRangeKernel (swap_rows I)");
            clFinish(command_queue);
        }

        // 3. Normalizar linha do pivô
        err = clSetKernelArg(kernels[3], 0, sizeof(cl_mem), &a_mem_obj);
        err |= clSetKernelArg(kernels[3], 1, sizeof(cl_mem), &i_mem_obj);
        err |= clSetKernelArg(kernels[3], 2, sizeof(int), &n);
        err |= clSetKernelArg(kernels[3], 3, sizeof(int), &k);
        checkError(err, "clSetKernelArg (normalize_row)");

        size_t global_norm = round_up(n, 256);
        size_t local_norm = 256;
        err = clEnqueueNDRangeKernel(command_queue, kernels[3], 1, NULL, &global_norm, &local_norm, 0, NULL, NULL);
        checkError(err, "clEnqueueNDRangeKernel (normalize_row)");
        clFinish(command_queue);

        // 4. Eliminação gaussiana
        err = clSetKernelArg(kernels[4], 0, sizeof(cl_mem), &a_mem_obj);
        err |= clSetKernelArg(kernels[4], 1, sizeof(cl_mem), &i_mem_obj);
        err |= clSetKernelArg(kernels[4], 2, sizeof(int), &n);
        err |= clSetKernelArg(kernels[4], 3, sizeof(int), &k);
        checkError(err, "clSetKernelArg (eliminate_row)");

        err = clEnqueueNDRangeKernel(command_queue, kernels[4], 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
        checkError(err, "clEnqueueNDRangeKernel (eliminate_row)");
        clFinish(command_queue);
    }

    double end_time = wtime();
    printf("Tempo de execução OpenCL: %.6f segundos\n", end_time - start_time);

    // Verificação
    if (n <= 5000) {
        err = clSetKernelArg(kernels[5], 0, sizeof(cl_mem), &a_original_mem_obj);
        err |= clSetKernelArg(kernels[5], 1, sizeof(cl_mem), &i_mem_obj);
        err |= clSetKernelArg(kernels[5], 2, sizeof(cl_mem), &result_mem_obj);
        err |= clSetKernelArg(kernels[5], 3, sizeof(int), &n);
        checkError(err, "clSetKernelArg (verify_result)");

        err = clEnqueueNDRangeKernel(command_queue, kernels[5], 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
        checkError(err, "clEnqueueNDRangeKernel (verify_result)");
        clFinish(command_queue);

        err = clEnqueueReadBuffer(command_queue, result_mem_obj, CL_TRUE, 0, n * n * sizeof(double), result, 0, NULL, NULL);
        checkError(err, "clEnqueueReadBuffer (resultado)");

        int valid = 1;
        double tolerance = 1e-4;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                if (fabs(result[i * n + j] - expected) > tolerance) {
                    valid = 0;
                    printf("Erro na posição [%d,%d]: %.6f vs %.1f\n", i, j, result[i * n + j], expected);
                    break;
                }
            }
            if (!valid) break;
        }
        printf("Verificação: %s\n", valid ? "SUCESSO" : "FALHA");
    }

    // Liberar recursos
    for (int i = 0; i < 6; i++) clReleaseKernel(kernels[i]);
    clReleaseProgram(program);
    clReleaseMemObject(a_mem_obj);
    clReleaseMemObject(i_mem_obj);
    clReleaseMemObject(result_mem_obj);
    clReleaseMemObject(pivot_vals_mem);
    clReleaseMemObject(a_original_mem_obj);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    free(A);
    free(I);
    free(result);
    free(pivot_vals);

    return 0;
}
