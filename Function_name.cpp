#include "Headers.h"
using namespace std;

namespace activation_functions
{
    double step_function(double u)
    {
        if (u > 1) return 1;
        else return 0;

    };
    double logistic_function(double u)
    {
        return 1/(1 + pow(M_E,-u));
    };

    double th(double u)
    {
        return (pow(M_E,u) - pow(M_E,-u)) / (pow(M_E,u) + pow(M_E,-u));
    };
    double linear (double u)
    {
        return u;
    };
    double d_th(double u)
    {
        return 4/pow( pow(M_E,u) + pow(M_E,-u), 2);
    };
}
namespace helpful_functions
{
    double target_function(vector<double> a, vector<double> target)
    {
        if (a.size() != target.size()) {cout<<"Demention error!"<<endl; return 0;}
        double T_F = 0;
        for (int i = 0; i < a.size(); i++)
            T_F += (a[i]-target[i])*(a[i]-target[i]);
        return T_F/2;
    }
}

