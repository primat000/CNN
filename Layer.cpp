#include "Headers.h"

using namespace std;

layer::layer(int n)
{
    size_l = n;
    list_neurons.resize(size_l);
}
void layer::resize(int n)
{
    list_neurons.resize(n);
    size_l = n;
}
int layer::size()
{
    return size_l;
}
neuron & layer::operator[](int n)
{
    return list_neurons[n];
}
