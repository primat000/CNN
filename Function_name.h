#ifndef FUNCTION_NAME_H_INCLUDED
#define FUNCTION_NAME_H_INCLUDED
#include <cmath>

namespace activation_functions
{
    double step_function(double u);
    double d_step_function(double u);
    double logistic_function(double u);
    double d_logistic_function(double u);
    double th(double u);
    double d_th(double u);
    double linear (double u);
};
namespace helpful_functions
{
    double target_function(vector<double> a, vector<double> target);
};

#endif // FUNCTION_NAME_H_INCLUDED
