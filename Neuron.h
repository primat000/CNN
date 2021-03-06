#ifndef NEURON_H_INCLUDED
#define NEURON_H_INCLUDED
#include "Function_name.h"
using namespace activation_functions;
/*................class map...............*/
class maps
{
public:
    vector<vector<double> > map;
    maps(const double* mas,int n,int m);
    maps(vector<vector<double> > mm);
    maps();
    vector<double> & operator[](int n);
    const maps& operator = (const maps &M);
    void map_resize(int n, int m);
    int n;
    int m;

};

/*................class neuron...............*/

class neuron
{
    vector<double> weights;
    vector<maps*> inputs;
    double (* function_type) (double);
    vector<vector<double> > feature_map;
    vector<vector<double> > final_feature_map;
    double EXIT;
    maps exit;
    int Kernel;
    int rec_n;
    int rec_m;
    int sh_n;
    int sh_m;
    int step;
    double w0;
public:
    neuron(double (*F_T)(double) = th, int r_n = 1, int r_m = 1, vector<maps*> M = {}, int shift_n = 1, int shift_m = 1, int Kernel = 1);
    void convolution ();
    maps* get_exit();
    void activate();//��� ������������
    vector<maps*> & get_inputs();
    void weight_ini();
    double function(double x);
    void change_function(double (*f)(double));
    void change_inputs(vector<maps*> new_inputs);
    double get_EXIT ();
    int count_weights();
    int feature_map_n();
    int feature_map_m();
    int inputs_n(int i);
    int inputs_m(int i);
    void change_weights(int n, double dw);
    void change_zero_weights(double dw);
    vector<vector<double> > Sigma;//������� ����, ������� �������� �� ����
    int get_rec_n();
    int get_rec_m();
    double get_weight(int n);
    double get_zero_weight();
    void info_inputs();
    int count_of_maps();
    int get_shift_n();
    int get_shift_m();
    void info_feature();

};



#endif // NEURON_H_INCLUDED
