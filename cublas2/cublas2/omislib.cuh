/*
*   Matrix structure that contains the matrix and its size. //optimizing memory access.
*/
struct mat {
    float* val;   // The values.
    int size[2];    // size[0] # of rows, size[1] # of cols.
};

struct vec {
    float* val;   // The values.
    int size;
};

void productMatVec(struct vec* c, struct mat* a, struct vec* b);
void productMatMat(struct mat* c, struct mat* a, struct mat* b);