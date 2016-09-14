#include "Headers.h"

using namespace std;

layer::layer(int n , double (*F_T)(double), int r_n, int r_m, vector<maps*> M, int shift_n, int shift_m, int K_s)
{

    size_l = n;
    //cout<<"\ncostructor size_l = "<<size_l<<endl;
    //cout<<r_n<<" "<<r_m<<" "<<shift_n<<" "<<shift_m<<endl;
    Kernel_size = K_s;
    if (Kernel_size > 1) KERNEL = true;
    else KERNEL = false;
    OUTS.resize(size_l);
    list_neurons.resize(size_l);
    //cout<<list_neurons.size()<<endl;
    for (int i = 0; i < size_l; i++)
    {
        neuron temp_n(F_T, r_n, r_m, M, shift_n, shift_m);
        //temp_n.info_inputs();
        list_neurons[i] = temp_n;
        list_neurons[i].activate();
        list_neurons[i].get_exit();
        //cout<<"\n exit neuron "<<i<<" :"<<endl;
        //list_neurons[i].info_feature();
    }

};

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
vector<maps*> & layer::Get_Outs()
{
    if (KERNEL)
    {
        for (int i = 0; i < list_neurons.size(); i++)
        {
            maps = list_neurons[i].get_exit();

        }
    }
    for (int i = 0; i < list_neurons.size(); i++)//создаем нейроны
    {
        OUTS[i] = list_neurons[i].get_exit();
    }
    return OUTS;
}
layer & layer::operator = (layer const& LAYER)
{
    KERNEL = LAYER.KERNEL;
    Kernel_size = LAYER.Kernel_size;
    size_l = LAYER.size_l;
    list_neurons = LAYER.list_neurons;
    OUTS = LAYER.OUTS;
    return *this;
}
void layer::info()
{
    cout<<"\n Layer Info"<<endl;
    cout<<"Count of neurons : "<<size_l<<endl;
    for (int i = 0; i < size_l; i++)
        {
            cout<<" "<<i<<" : "<<endl;
            list_neurons[i].info_inputs();
        }

    cout<<"\nWeights"<<endl;

    for (int j = 0; j < list_neurons.size(); j++)
        {
            cout<<"neuron : "<<j<<endl;
            cout<<"zero_weight = "<<list_neurons[j].get_zero_weight()<<endl;
            for(int q = 0; q < list_neurons[j].count_weights(); q++)
            {
                    cout<<list_neurons[j].get_weight(q)<<"  ";
            }
            cout<<endl;
        }

}
