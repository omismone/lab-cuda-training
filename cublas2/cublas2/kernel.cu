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
    for (int i = 0; i < m->size; i++) {
        printf("%.0f ", m->val[i]);
        //if (i != m->size - 1) printf(", ");
    }
    printf("\n");
}

/*
*   Generate a random matrix.
*
*   @param arr The matrix.
*/
void getRandomMat(struct mat* m) {
    for (int i = 0; i < m->size[0]; i++) {
        for (int j = 0; j < m->size[1]; j++) {
            m->val[i * m->size[1] + j] = (float)(rand() % 10);
        }
    }
}

/*
*   Print a linear matrix to stdout.
*
*   @param mat The matrix.
*/
void printMat(struct mat* m) {
    for (int i = 0; i < (m->size[0] * m->size[1]); i++) {
        if (i % m->size[1] == 0) printf("\n");
        printf("%.0f ", m->val[i]);
    }
    printf("\n");
}

/*
*   Calculate the transposed of a matrix.
*
*   @param mat The matrix.
*   @param res The place to save the result.
*/
void transpose(struct mat* mat, struct mat* res) {
    for (int i = 0; i < mat->size[0]; i++) {
        for (int j = 0; j < mat->size[1]; j++) {
            res->val[i * res->size[1] + j] = mat->val[j * res->size[1] + i];
        }
    }
}


int main()
{
    //init
    const int ROWS = 2;
    const int COLS = 3;
    struct mat a, d, e; //will perform a * b = c and a * d = e
    struct vec b, c;
    a.size[0] = ROWS;
    a.size[1] = COLS;    
    d.size[0] = COLS;
    d.size[1] = ROWS; 
    e.size[0] = ROWS;
    e.size[1] = ROWS;

    b.size = COLS;
    c.size = ROWS;

    a.val = (float*)malloc(a.size[0] * a.size[1] * sizeof(float));
    d.val = (float*)malloc(d.size[0] * d.size[1] * sizeof(float));
    e.val = (float*)malloc(e.size[0] * e.size[1] * sizeof(float));
    b.val = (float*)malloc(b.size * sizeof(float));
    c.val = (float*)malloc(c.size * sizeof(float));

    //populate with rand val
    srand(time(NULL));   // initialization
    getRandomMat(&a);
    getRandomVec(&b);
    getRandomMat(&d);

    productMatVec(&c, &a, &b); //level 2 function

    productMatMat(&e, &a, &d); //level 3 function

    //out 1
    printf("a)\n");
    printMat(&a);
    printf("b)\n");
    printVec(&b);

    printf("product)\n");
    printVec(&c);

    //out 2
    printf("\n\n\n\na)\n");
    printMat(&a);
    printf("d)\n");
    printMat(&d);

    printf("product)\n");

    struct mat res;
    res.size[0] = e.size[1];
    res.size[1] = e.size[0];
    res.val = (float*)malloc(res.size[0] * res.size[1] * sizeof(float));
    transpose(&e, &res);

    printMat(&res);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaDeviceReset();

    //free
    free(a.val);
    free(b.val);    
    free(c.val);    
    free(d.val);   
    free(e.val);

    return 0;
}