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
    double average_weight;
    net(maps& Image, vector<vector<int> > Info);
    net();
    net(const char* filename);
    int result(maps Image, bool flag = false);
    void back_propogation(vector<double> target, maps Image, double Speed = 0.01);
    void back_propogation_online(vector<double> target, maps Image, double Speed = 0.01);
    void back_propogation_offline(vector<vector<double> > &target, vector<maps> &Image, double Speed = 0.01, bool isForGen = false);
    void preparation_backprop();
    void info();
    void back_prop_info();
    void add_layer(int number, int count_neurons, int rec_n, int rec_m, int sh_n, int sh_m, int Ker);
    void add_neuron(int position_in_layer, int layers_number);
    void remove_layer(int number);
    void remove_neuron(int position_in_layer, int layers_number);
    void dementions_of_exits();
    void save_net(const char* filename);
    void activate(maps Image);
    net & operator = (const net & NET);
    layer & operator[](int n);
    inline int net_size() {return list_layers.size();};
    void shake(double n = -1);
    void genetic_algo(int iter, int count, double per_mutation, vector <maps> & Train, vector<vector<double> > &Targets);
    void zeroize_diff();
    void update_weights(double step);
    void cross (net & NET1, net & NET2);
};
template<typename T>
int pos_max (vector<T> & v);
template<typename T>
int pos_min (vector<T> & v);

#endif // NET_H_INCLUDED
