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
    void back_propogation(vector<double> target, maps Image);
    void info();
    void back_prop_info();

};

#endif // NET_H_INCLUDED
