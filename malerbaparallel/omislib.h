#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

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
*   Hippocampal pyramidal layer parameters
*/
struct pm{
    /* E = excitatory ; I = inhibitory */

    float CE;   // capacity [F]
    float glE;  // leakage conductance [S]
    float ElE;  // leakage reversal potential [V]
    float aE;   // constant [ ]
    float bE;   // constant [ ]
    float slpE; // 
    float twE;  // [s]
    float VtE;  //soglia
    float VrE;  //riposo
    
    float CI; 
    float glI;
    float ElI;
    float aI;
    float bI;
    float slpI;
    float twI;
    float VtI;
    float VrI;

    float gnoiseE;  //
    float gnoiseI;

    //EtoE
    float tauEr;
    float tauEd;
    //EtoB
    float tauEIr;
    float tauEId;
    //BtoB
    float tauIr;
    float tauId;
    //BtoP 
    float tauIEr;
    float tauIEd;

    float gmaxII;
    float gmaxEI;
    float gmaxIE;
    float gmaxEE;

    float VrevE;
    float VrevI;
    
    float gvarEE;
    float gvarII;

    float gvarEI;
    float gvarIE;

    float DCstdI;
    float DCstdE;

    float Edc;
    float jmpE;
    float Idc;
    float jmpI;

    float seqsize;
    float dcbias;
};

/*
*   input sequence
*/
struct inpseq{
    float slp;
    float *on;
    float length;
};

struct options{
    int nonoise; //if no noise added, turn to 1
    int novar; //if no variance in synaptic weightsm turn to 1
    int storecurrs; //if you want the output to include the synaptic currents
    float noiseprc; //percent of standard deviation of the noise to use in the simulation
    int seqassign; //if you want to choose 10 cells that are going to be part of a sequence
};

/*
*   neurons connections
*/
struct Conn{
    struct matrix EtoE;
    struct matrix ItoI;
    struct matrix EtoI;
    struct matrix ItoE;
};

/*
*   v_n dentr Isyn
*/
struct Vbar{
    double *E;
    double *I;
};

/*
*   outputs
*/
struct Veg{
    double *E;
    double *I;
    int ne;
    int ni;
};

/*
*  spikes times
*/
struct Tsp{
    double *times;
    double *celln;
};

/*
*   I_syn
*/
struct Isynbar{
    struct matrix ItoE;
    struct matrix EtoE;
};

/*
*   don't need this if all neurons have the same inputs
*/
struct Inp{
    double *Edc;
    double *Idc;
    struct matrix Etrace;
    struct matrix Itrace;
};

/*
*   the function that runs the simulation
*/
void NetworkRunSeqt(struct pm p, struct inpseq in, int NE, int NI, float T, struct options opt);

