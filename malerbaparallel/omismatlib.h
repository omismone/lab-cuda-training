#include<stdlib.h>
#include<math.h>

/*
*   Array structure that contains the array and its size.
*/
struct vec{
    double *val;   // The array.
    int size;    // Number of its elements.
};

/*
*   Matrix structure that contains the matrix and its size. //optimizing memory access.
*/
struct matrix{
    double *val;   // The values.
    int size[2];    // size[0] # of rows, size[1] # of cols.
};

/*  
*   r = j:i:k regularly-spaced vector
*
*   @attention when done call free(r.val)
*/
struct vec *regSpaceVec(double j, double i, double k){
    struct vec *r;
    if((i==0) || (i>0 && j>k) || (i<0 && j<k))
        return r;
    int s = ceil(k/i);
    r->size = s;
    r->val = malloc(sizeof(double) * r->size);
    for(int idx=0; i * idx <= k; idx++){
        r->val[idx] = i * idx; 
    }
}

/*
*   @see https://en.wikipedia.org/wiki/Normal_distribution#Generating_values_from_normal_distribution
*/
double RandNormal() 
{
    double x = (double)rand() / (double)RAND_MAX;
    double y = (double)rand() / (double)RAND_MAX;
    double z = sqrt(-2 * log(x)) * cos(2 * 3.14159265358979323846 * y);
    
    return z;
}