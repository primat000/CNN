#include <iostream>
#include "Headers.h"
#include <cstdio>
#include <stdio.h>
#include <fstream>


using namespace std;
double f(double x);
template <typename T, int N>
vector<T> magic(T (&v) [N]);//из массива в вектор


int main()
{
    srand(time(0));
    int count = 500;
    ifstream fileTarget ( "result.txt" );
    ifstream fileImage ( "My digits.txt" );
    //work

    vector<vector<int> > Info1(4);
    vector<vector<int> > Info2(3);

    vector <vector <double> > temp_image(8);
    for (int i = 0; i < 8; i++)
        temp_image[i].resize(8);
    vector<double> temp_target(10,0);
    vector <maps> Inputs(count);
    vector<vector<double> > Targets(count);
    for (int i = 0; i < count; i++)
    {
        for (int k = 0; k < 8; k++)
        {
            for (int m = 0; m < 8; m++)
            {
                fileImage>>temp_image[k][m];
                //cout<<temp_image[k][m]<<" ";
            }
            //cout<<endl;
        }

        maps map_im(temp_image);
        Inputs[i] = map_im;
        int a;
        fileTarget>>a;
        //cout<<"Target = "<<a<<"\t";
        Targets[i] = temp_target;
        Targets[i][a] = 1;
    }
    // create net
    int temp1 [6] = {4,2,2,2,2,1};
    Info1[0] = magic(temp1);
    int temp2 [6] = {6,2,2,2,2,1};
    Info1[1] = magic(temp2);
    int temp3 [6] = {10,2,2,1,1,1};
    Info1[2] = magic(temp3);
    int temp4 [6] = {10,1,1,1,1,1};
    Info1[3] = magic(temp4);

    int temp12 [6] = {4,2,2,2,2,1};
    Info2[0] = magic(temp12);
    int temp22 [6] = {6,2,2,2,2,1};
    Info2[1] = magic(temp22);
    int temp32 [6] = {10,2,2,1,1,1};
    Info2[2] = magic(temp32);

    vector<vector<int> > Info3(3);
    int temp13 [6] = {6,4,4,4,4,1};
    Info3[0] = magic(temp13);
    int temp23 [6] = {10,2,2,1,1,1};
    Info3[1] = magic(temp23);
    int temp33 [6] = {10,1,1,1,1,1};
    Info3[2] = magic(temp33);


    vector<vector<int> > Info4(2);
    int temp14[6] = {6,4,4,4,4,1};
    Info4[0] = magic(temp14);
    int temp24 [6] = {10,2,2,1,1,1};
    Info4[1] = magic(temp24);

    vector<vector<int> > Info5(5);
    int temp15 [6] = {4,4,4,4,4,1};
    Info5[0] = magic(temp15);
    int temp25 [6] = {10,2,2,1,1,1};
    Info5[1] = magic(temp25);
    int temp35 [6] = {6,1,1,1,1,1};
    Info5[2] = magic(temp35);
    int temp45 [6] = {5,1,1,1,1,1};
    Info5[3] = magic(temp45);
    int temp55 [6] = {10,1,1,1,1,1};
    Info5[4] = magic(temp55);

    //net Temp_Best;
    /*net CNN1(Inputs[0], Info1);
    net CNN2(Inputs[1], Info2);
    net CNN3(Inputs[1], Info3);
    net CNN4 (Inputs[1], Info4);*/
    //net CNN5 (Inputs[1], Info5);

    /*net CNN1("Best_1.txt");
    CNN1.activate(Inputs[1]);
    CNN1.preparation_backprop();
    cout<<CNN1.result(Inputs[1])<<endl;
    net CNN2("Best_2.txt");
    CNN2.activate(Inputs[1]);
    CNN2.preparation_backprop();
    cout<<CNN2.result(Inputs[1])<<endl;
    net CNN3("Best_3.txt");
    CNN3.activate(Inputs[1]);
    CNN3.preparation_backprop();
    cout<<CNN3.result(Inputs[1])<<endl;
    net CNN4 ("Best_4.txt");
    CNN4.activate(Inputs[1]);
    CNN4.preparation_backprop();
    cout<<CNN4.result(Inputs[1])<<endl;*/
    net CNN3 (Inputs[1], Info3);
    CNN3.genetic_algo(50, 5, 0.2, Inputs, Targets);
    /*net CNN3 ("Best_3.txt");
    CNN3.activate(Inputs[1]);
    CNN3.preparation_backprop();
    cout<<CNN3.result(Inputs[1])<<endl;*/

    double max1 = 0, global_max1 = 0;
    double max2 = 0, global_max2 = 0;
    double max3 = 0, global_max3 = 0;
    double max4 = 0, global_max4 = 0;
    double max5 = 0, global_max5 = 0;

    double step1 = 0.01, step2 = 0.01, step3 = 0.01, step4 = 0.01, step5 = 0.05;

    cout<<"!!!! offline !!!!"<<endl;
    vector<maps> im(10);
    vector<vector<double> > tar(10);
    /*for (int i = 0; i < 0; i++)
    {
        max1 = 0; max2 = 0; max3 = 0; max4 = 0; max5 = 0;

         for (int j = 0; j < 10; j++)
        {
            step1 = 0.01; step2 = 0.01; step3 = 0.01; step4 = 0.01; step5 = 0.01;
            int random = (int) (rand() % (Targets.size() -201) +200);
            im[j] = Inputs[random];
            tar[j] = Targets[random];
        }
        CNN5.back_propogation_offline(tar, im, step3);
        for (int j = 0; j < 200; j++)
        {
            if (CNN3.result(Inputs[j])==pos_max(Targets[j])) max3++;
        }
        if (global_max3 < max3) {global_max3 = max3; CNN3.save_net("Best_3.txt");}
        cout<<"\nMax5 = "<<max5<<endl;
        cout<<"GlobalMax5 = "<<global_max3/200<<endl;
    }*/
    return 0;
}
double f(double x)
{
    return x;
}
template <typename T, int N>
vector<T> magic(T (&v) [N])
{
    vector<T> temp(N);
    for (int i = 0; i < N; i++)
        temp[i] = v[i];
        return temp;
}


