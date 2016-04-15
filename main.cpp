#include <iostream>
#include "Headers.h"

using namespace std;
double f(double x);

int main()
{
    double my[3][3] = {{0,1,1},{1,1,1},{1,1,1}};
    double mmy[3][3] = {{1,1,1},{1,0,1},{1,1,1}};
    maps m1(&my[0][0],3,3),m2(&mmy[0][0],3,3);
    vector<maps> Maps;
    Maps.push_back(m1);
    double Im1[4][4] = {{1,1,1,1},{1,0,0,1},{1,0,0,1},{1,1,1,1}};
    double Im2[4][4] = {{0,0,0,0},{0,1,1,0},{0,1,1,0},{0,0,0,0}};
    double Test[4][4] = {{1,1,1,1},{1,0,0,1},{1,0,0,1},{1,0,1,1}};
    //double Im[3][3] = {{1,1,1},{1,0,0},{1,0,0}};
    maps IIm1(&Im1[0][0],4,4);
    maps IIm2(&Im2[0][0],4,4);
    maps TTest(&Test[0][0],4,4);
    /*Maps.push_back(m2);
    neuron n(f, 2, 2,Maps);
    neuron nn(f, 3, 3,Maps);
    n.activate();
    n.get_exit();
    nn.activate();
    nn.get_exit();
    return 0;
    int Inf[1][4] = {2,3,2,2};
    maps Info(&Inf[0][0],1,4);*/
    vector<double> V1(2,1),V2(2,1);
    V1[0] = 0.5;
    V1[1] = 0;
    V2[0] = 0;
    V2[1] = 0.5;
    vector<vector<int> > Info(3);
    Info[0].push_back(2);
    Info[0].push_back(3);
    Info[0].push_back(2);
    Info[0].push_back(2);
    Info[1].push_back(1);
    Info[1].push_back(3);
    Info[1].push_back(2);
    Info[1].push_back(2);
    Info[2].push_back(2);
    Info[2].push_back(3);
    Info[2].push_back(2);
    Info[2].push_back(2);
    /*vector<vector<int> > Info(2);
    Info[0].push_back(2);
    Info[0].push_back(3);
    Info[0].push_back(2);
    Info[0].push_back(2);
    Info[1].push_back(2);
    Info[1].push_back(3);
    Info[1].push_back(2);
    Info[1].push_back(2);*/
    net CNN(IIm1, Maps, Info);
    for (int i = 0; i < 190; i++)
    {
        CNN.back_propogation(V1,IIm1);
        //CNN.result(IIm1);
        CNN.back_propogation(V2,IIm2);
        //CNN.result(IIm2);
    }
    cout<<"\n"<<endl;
    CNN.result(IIm1);
    CNN.info();
    CNN.result(TTest);
    CNN.info();
}
double f(double x)
{
    return x;
}
