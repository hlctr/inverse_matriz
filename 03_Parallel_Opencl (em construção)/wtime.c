#include <sys/time.h>
#include <stdlib.h>

// Função para medir o tempo em segundos
double wtime(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}
