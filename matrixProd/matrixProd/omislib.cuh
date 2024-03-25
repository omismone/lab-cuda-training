/*
*   Matrix structure that contains the matrix and its size. //optimizing memory access.
*/
struct mat {
    int *val;   // The values.
    int size[2];    // size[0] # of rows, size[1] # of cols.
};

void matrixMul(struct mat* c, struct mat* a, struct mat* b);