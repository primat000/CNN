#ifndef LAYER_H_INCLUDED
#define LAYER_H_INCLUDED

class layer
{
    vector<neuron> list_neurons;
    int size_l;
public:
    layer(int n = 0);
    void resize(int n);
    int size();
    neuron & operator[](int n);
    vector<double> OUT(vector<double> set);
    vector<double> Inputs();
};

#endif // LAYER_H_INCLUDED
