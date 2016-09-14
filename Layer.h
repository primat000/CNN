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
public:
    layer(int n = 0, double (*F_T)(double) = th, int r_n = 1, int r_m = 1, vector<maps*> M = {}, int shift_n = 1, int shift_m = 1, int K_s = 1);
    void resize(int n);
    int size();
    neuron & operator[](int n);
    //vector<double> OUT(vector<double> set);
    vector<double> Inputs();
    vector<maps*> & Get_Outs();
    layer & operator = (layer const& LAYER);
    void info();
};

#endif // LAYER_H_INCLUDED
