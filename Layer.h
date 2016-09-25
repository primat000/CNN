#ifndef LAYER_H_INCLUDED
#define LAYER_H_INCLUDED
#include "Function_name.h"
using namespace activation_functions;
class layer
{
    bool KERNEL;
    int Kernel_size;
    vector<neuron> list_neurons;
    vector<maps*> OUTS;
    int size_l;
    bool Act = false;
public:
    layer(int n = 0, double (*F_T)(double) = th, int r_n = 1, int r_m = 1, vector<maps*> M = {}, int shift_n = 1, int shift_m = 1, int K_s = 1);
    layer(int n, int count_maps,int size_image_n, int size_image_m, int r_n, int r_m, int shift_n, int shift_m, int K,  double (*F_T)(double) = th);
    void resize(int n);
    int size();
    neuron & operator[](int n);
    //vector<double> OUT(vector<double> set);
    vector<double> Inputs();
    vector<maps*> & Get_Outs();
    layer & operator = (const layer & LAYER);
    void info();
    inline bool is_reduce(){return KERNEL;};

};

#endif // LAYER_H_INCLUDED
