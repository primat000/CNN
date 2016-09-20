#ifndef NET_H_INCLUDED
#define NET_H_INCLUDED
#include <fstream>
class net
{

    vector<layer> list_layers;
    maps Image;
    vector<double> exit;
    vector<vector <maps> > d_outs;
    vector<vector <vector <double> > > d_weights;
    vector<vector <double> > zero_d_weights;
    vector<vector <maps> > d_reduce_outs;

    int n; // количество слоев
public:
    net(maps Image, vector<vector<int> > Info);
    net();
    net(const char* filename);
    int result(maps Image, bool flag = false);
    void back_propogation(vector<double> target, maps Image, double Speed = 0.3);
    void preparation_backprop();
    void info();
    void back_prop_info();
    void add_layer(int number, int count_neurons);
    void add_neuron(int position_in_layer, int layers_number);
    void remove_layer(int number);
    void remove_neuron(int position_in_layer, int layers_number);
    void dementions_of_exits();
    void save_net(const char* filename);
    void activate(maps Image);
    net & operator = (net const& NET);
    layer & operator[](int n);
    inline int net_size() {return list_layers.size();};
};

#endif // NET_H_INCLUDED
