#include <iostream>
#include "Headers.h"
#include <cstdio>
#include <stdio.h>

using namespace std;
double f(double x);
vector<int> magic(int *v, int n);//из массива в вектор

int main()
{
    srand(time(0));
    double my[3][3] = {{0,1,1},{1,1,1},{1,1,1}}; //
    double mmy[3][3] = {{1,1,1},{1,0,1},{1,1,1}}; // conection
    maps m1(&my[0][0],3,3),m2(&mmy[0][0],3,3);
    vector<maps> Maps;
    Maps.push_back(m1);
    //work

    vector<vector<int> > Info(3);

    double Im1[4][4] = {{1,1,1,1},{1,0,0,1},{1,0,0,1},{1,1,1,1}};
    double Im2[4][4] = {{0,0,0,0},{0,1,1,0},{0,1,1,0},{0,0,0,0}};
    double Im3[4][4] = {{1,0,0,1},{0,1,1,0},{0,1,1,0},{1,0,0,1}};
    double Im4[4][4] = {{0,0,0,1},{0,0,1,0},{0,1,0,0},{1,0,0,0}};
    double Test[4][4] = {{0,0,0,0},{0,0,1,0},{0,1,0,0},{1,0,0,0}};
    maps IIm1(&Im1[0][0],4,4);
    maps IIm2(&Im2[0][0],4,4);
    maps IIm3(&Im3[0][0],4,4);
    maps IIm4(&Im4[0][0],4,4);
    maps ImTest(&Test[0][0],4,4);
    vector<double> V1(4),V2(4),V3(4),V4(4); // target
    V1[0] = 1;
    V1[1] = 0;
    V1[2] = 0;
    V1[3] = 0;
    V2[0] = 0;
    V2[1] = 1;
    V2[2] = 0;
    V2[3] = 0;
    V3[0] = 0;
    V3[1] = 0;
    V3[2] = 1;
    V3[3] = 0;
    V4[0] = 0;
    V4[1] = 0;
    V4[2] = 0;
    V4[3] = 1;
    Info[0].push_back(2);
    Info[0].push_back(2);
    Info[0].push_back(2);
    Info[0].push_back(1);
    Info[0].push_back(1);
    Info[0].push_back(1);
    Info[0].push_back(1);
    Info[1].push_back(1);
    Info[1].push_back(2);
    Info[1].push_back(2);
    Info[1].push_back(1);
    Info[1].push_back(1);
    Info[1].push_back(1);
    Info[1].push_back(1);
    Info[2].push_back(4);
    Info[2].push_back(2);
    Info[2].push_back(2);
    Info[2].push_back(1);
    Info[2].push_back(1);
    Info[2].push_back(1);
    Info[2].push_back(1);



    //net CNN1(IIm1, Maps, Info);
    //CNN1.info();

    //CNN1.result(IIm4);

    /*net CNN2(IIm1, Maps, Info);
    CNN2.info();

    CNN1.result(IIm4,1);
    CNN2.result(IIm1,1);
    for (int i = 0; i < 100; i++)
    {
        CNN1.back_propogation(V1,IIm1, 0.1);
        CNN1.back_propogation(V2,IIm2, 0.1);
        CNN1.back_propogation(V3,IIm3, 0.1);
        CNN1.back_propogation(V4,IIm4, 0.1);
        CNN2.back_propogation(V1,IIm1, 0.1);
        CNN2.back_propogation(V2,IIm2, 0.1);
        CNN2.back_propogation(V3,IIm3, 0.1);
        CNN2.back_propogation(V4,IIm4, 0.1);
    }

    CNN1.result(IIm4,1);
    CNN2.result(IIm1,1);*/
    vector<maps*> Im;
    Im.push_back(&IIm1);
    layer TestLayer(2,th, 2, 2, Im, 1, 1, 4);
    TestLayer.info();

    //CNN1.info();

    return 0;
}
double f(double x)
{
    return x;
}
vector<int> magic(int v[], int n)
{
    vector<int> temp(n);
    for (int i = 0; i < n; i++)
        temp[i] = v[i];
        return temp;
}
