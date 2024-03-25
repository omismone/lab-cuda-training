/*
*   Array structure that contains the array and its size.
*/
struct arr {
    int* val;   // The array.
    int size;    // Number of its elements.
};

void addWithCuda(struct arr *c, struct arr *a, struct arr *b);
int dotProdWithCuda(struct arr* a, struct arr* b);