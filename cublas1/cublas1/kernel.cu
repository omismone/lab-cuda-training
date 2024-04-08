
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <time.h>
#include<stdio.h>

#include "omislib.cuh"


/*
*   Generate a random array.
*
*   @param arr The array.
*/
void getRandomVec(struct vec* m) {
    for (int i = 0; i < m->size; i++) {
        m->val[i] = (float)(rand() % 10);
    }
}

/*
*   Print a linear array to stdout.
*
*   @param vec The array.
*/
void printVec(struct vec* m) {
    printf("[");
    for(int i = 0; i < m->size; i++) {
        printf("%.0f", m->val[i]);
        if (i % m->size-1 != 0) printf(", ");
    }
    printf("]\n");
}


int main()
{
    //init
    const int SIZE = 2;
    struct vec a;
    struct vec b;
    a.size = SIZE;
    b.size = SIZE;

    a.val = (float*)malloc(a.size * sizeof(float));
    b.val = (float*)malloc(b.size * sizeof(float));

    float dotprod;

    //populate a and b with rand val
    srand(time(NULL));   // initialization
    getRandomVec(&a);
    getRandomVec(&b);

    // Add arrtors in parallel.

    dotProduct(&dotprod, &a, &b);

    //out
    printf("a)\n");
    printVec(&a);
    printf("b)\n");
    printVec(&b);

    printf("dot product)    %.0f\n", dotprod);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaDeviceReset();

    //free
    free(a.val);
    free(b.val);

    return 0;
}
