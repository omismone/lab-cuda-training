
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "omislib.cuh"


/*
*   Generate a random matrix.
*
*   @param arr The matrix.
*/
void getRandomMat(struct mat *m) {
    for (int i = 0; i < m->size[0]; i++) {
        for (int j = 0; j < m->size[1]; j++) {
            m->val[i*m->size[1] + j] = rand() % 10;
        }
    }
}

/*
*   Print a linear matrix to stdout.
*
*   @param mat The matrix.
*/
void printMatrix(struct mat* m) {
    for (int i = 0; i < (m->size[0] * m->size[1]); i++) {
        if (i % m->size[1] == 0) printf("\n");
        printf("%d ", m->val[i]);
    }
    printf("\n");
}


int main()
{
    //init
    const int SIZE = 3;
    struct mat a;
    struct mat b;
    a.size[0] = SIZE;
    a.size[1] = SIZE;
    b.size[0] = SIZE;
    b.size[1] = SIZE;
    a.val = (int*)malloc(a.size[0] * a.size[1] * sizeof(int));
    b.val = (int*)malloc(b.size[0] * b.size[1] * sizeof(int));

    struct mat c;
    c.size[0] = SIZE;
    c.size[1] = SIZE;
    c.val = (int*)malloc(c.size[0] * c.size[1] * sizeof(int));

    //populate a and b with rand val
    srand(time(NULL));   // initialization
    getRandomMat(&a);
    getRandomMat(&b);

    // Add arrtors in parallel.
    matrixMul(&c, &a, &b);

    //out
    printf("\na)");
    printMatrix(&a);
    printf("\nb)");
    printMatrix(&b);
    printf("\nprod)");
    printMatrix(&c);


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaDeviceReset();

    //free
    free(a.val);
    free(b.val);
    free(c.val);

    return 0;
}
