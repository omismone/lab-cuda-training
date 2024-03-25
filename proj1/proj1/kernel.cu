
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "omislib.cuh"


/*
*   Generate a random array.
*
*   @param arr The array.
*/
void getRandomArr(struct arr *v) {
    for (int i = 0; i < v->size; i++) {
        v->val[i] = rand() % 10;
    }
}

int main()
{
    //init
    const int SIZE = 3;
    struct arr a;
    struct arr b;
    a.size = SIZE;
    b.size = SIZE;
    a.val = (int *)malloc(SIZE * sizeof(int));
    b.val = (int *)malloc(SIZE * sizeof(int));

    struct arr c;
    c.size = SIZE;
    c.val = (int*)malloc(SIZE * sizeof(int));

    //populate a and b with rand val
    srand(time(NULL));   // initialization
    getRandomArr(&a);
    getRandomArr(&b);

    // Add arrtors in parallel.
    addWithCuda(&c, &a, &b);

    // Dot product "in parallel"
    int dot = dotProdWithCuda(&a, &b);

    //out

    //add
    printf("{");
    for (int i = 0; i < a.size; i++) {
        printf("%d", a.val[i]);
        if (i != a.size - 1) printf(", ");
    }
    printf("} + {");
    for (int i = 0; i < b.size; i++) {
        printf("%d", b.val[i]);
        if (i != b.size - 1) printf(", ");
    }
    printf("} = {");
    for (int i = 0; i < c.size; i++) {
        printf("%d", c.val[i]);
        if (i != c.size - 1) printf(", ");
    }
    printf("}\n");

    //dot prod
    printf("dot product: %d\n", dot);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaDeviceReset();

    //free
    free(a.val);
    free(b.val);
    free(c.val);

    return 0;
}
