#ifndef NET_H_INCLUDED
#define NET_H_INCLUDED


class net
{

    vector<layer> list_layers;
    maps Image;
    vector<double> exit;
    vector<maps> connection;
    vector<vector<maps> > d_outs;
    vector<vector<vector<double> > > d_weights;
    int n; // количество слоев
public:
    net(maps Image, vector<maps> Con , vector<vector<int> > Info);
    void result(maps Image);
    void back_propogation(vector<double> target, maps Image, double Speed = 0.4);
    void info();
    void back_prop_info();
    void add_layer(int number, int count_neurons);
    void add_neuron(int position_in_layer, int layers_number);
    void remove_layer(int number);
    void remove_neuron(int position_in_layer, int layers_number);
};

#endif // NET_H_INCLUDED
