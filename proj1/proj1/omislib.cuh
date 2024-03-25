/*
*   Array structure that contains the array and its size.
*/
struct arr {
    int* val;   // The array.
    int size;    // Number of its elements.
};

int addWithCuda(struct arr *c, struct arr *a, struct arr *b);