#include <iostream>
#include "Headers.h"
#include <cstdio>
#include <stdio.h>

using namespace std;
double f(double x);
vector<int> magic(int *v, int n);//из массива в вектор

int main()
{
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
    V1[0] = 0.5;
    V1[1] = 0;
    V1[2] = 0;
    V1[3] = 0;
    V2[0] = 0;
    V2[1] = 0.5;
    V2[2] = 0;
    V2[3] = 0;
    V3[0] = 0;
    V3[1] = 0;
    V3[2] = 0.5;
    V3[3] = 0;
    V4[0] = 0;
    V4[1] = 0;
    V4[2] = 0;
    V4[3] = 0.5;
    Info[0].push_back(5);
    Info[0].push_back(3);
    Info[0].push_back(2);
    Info[0].push_back(2);
    Info[1].push_back(1);
    Info[1].push_back(3);
    Info[1].push_back(2);
    Info[1].push_back(2);
    Info[2].push_back(4);
    Info[2].push_back(3);
    Info[2].push_back(2);
    Info[2].push_back(2);


    /* i dont know
    double Im1[8][6] = {{0,0,0,0,0,0},{0,0,0.5,1,0,0},{0,0.5,1,1,0,0},{0,1,0,1,0,0},{0,0,0,1,0,0},{0,0,0,1,0,0},{0,0,0,0,0,0},{0,0,0,0,0,0}};
    double Im2[8][6] = {{0,0,0,0,0,0},{0,0.5,1,1,1,0},{0,1,0,0,1,0},{0,0,0,0.5,1,0},{0,0,1,1,0,0},{0,0,1,0,0,0},{0,0,1,1,1,1},{0,0,0,0,0,0}};
    double Im3[8][6] = {{0,0,0,0,0,0},{0,1,1,1,1,0},{0,0,0,0,1,0},{0,0,0,1,0,0},{0,0,1,0,0,0},{0,0,1,1,0,0},{0,0,0,0,1,0},{0,1,1,1,1,0}};
    maps IIm1(&Im1[0][0],8,6);
    maps IIm2(&Im2[0][0],8,6);
    maps IIm3(&Im3[0][0],8,6);
    vector<double> V1(3),V2(3),V3(3); // target
    V1[0] = 0.5;
    V1[1] = 0;
    V1[2] = 0;
    V2[0] = 0;
    V2[1] = 0.5;
    V2[2] = 0;
    V3[0] = 0;
    V3[1] = 0;
    V3[2] = 0.5;

    int L1[4] = {2,3,3,2};
    int L2[4] = {1,3,3,2};
    int L3[4] = {1,3,3,3};
    int L4[4] = {3,3,2,2};
    Info[0] = magic(L1,4);
    Info[1] = magic(L2,4);
    Info[2] = magic(L3,4);
    Info[3] = magic(L4,4);*/


    /*work
    net CNN(IIm1, Maps, Info);
    CNN.result(IIm1);
    CNN.info();
    for (int i = 0; i < 500; i++)
    {
        CNN.back_propogation(V1,IIm1);
        CNN.back_propogation(V2,IIm2);
        CNN.back_propogation(V3,IIm3);
        CNN.back_propogation(V4,IIm4);
        //CNN.result(IIm1);
        //CNN.back_propogation(V3,IIm3);
    }
    cout<<"\n "<<endl;
    cout<<"\n "<<endl;
    CNN.result(IIm1);
    cout<<"\n "<<endl;
    CNN.result(IIm2);
    cout<<"\n "<<endl;
    CNN.result(IIm3);
    cout<<"\n "<<endl;
    CNN.result(IIm4);
    cout<<"\n "<<endl;
    CNN.result(ImTest);

    CNN.info();*/
    //test
    /*double Im1[6][6] = {{-1,0,0,0,0,1},{1,0,0,0,0,0},{0,1,-1,0,0,1},{1,1,0,0,0,0},{1,0,1,0,0,1},{0,0,1,0,1,0}};
    maps IIm1(&Im1[0][0],6,6);
    vector<double> V1(3); // target
    vector<vector<int> > Info(5);
    Info[0].push_back(5);////////
    Info[0].push_back(3);
    Info[0].push_back(2);
    Info[0].push_back(2);

    Info[1].push_back(2);////////////
    Info[1].push_back(3);
    Info[1].push_back(2);
    Info[1].push_back(2);

    Info[2].push_back(2);////////
    Info[2].push_back(1);
    Info[2].push_back(2);
    Info[2].push_back(2);

    Info[3].push_back(3);///////
    Info[3].push_back(3);
    Info[3].push_back(2);
    Info[3].push_back(2);

    Info[4].push_back(3);////////
    Info[4].push_back(3);
    Info[4].push_back(2);
    Info[4].push_back(2);
    V1[0] = 0.5;
    V1[1] = 0;
    V1[2] = 0;*/
    net CNN(IIm1, Maps, Info);
    CNN.result(IIm1);
    CNN.back_prop_info();
    CNN.info();
    for (int i = 0; i < 70; i++)
    {
        CNN.back_propogation(V1,IIm1, 0.3);
        CNN.back_propogation(V2,IIm2);
        CNN.back_propogation(V3,IIm3, 0.2);
        CNN.back_propogation(V4,IIm4);
    }

    CNN.back_prop_info();
    CNN.result(ImTest);
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
