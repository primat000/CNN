#include "Headers.h"
#include <cmath>
using namespace std;
using namespace activation_functions;

net::net(maps IMAGE, vector<maps> Con, vector<vector<int> > Info)
{

    n = Info.size();// количество слоев
    list_layers.resize(n);
    list_layers[0].resize(Info[0][0]);// создаем первый слой
    //cout<<"zero layer"<<endl;
    for (int i = 0; i < list_layers[0].size(); i++)//создаем нейроны первого слоя
    {
        vector<maps*> Im;
        Image = IMAGE;
        Im.push_back(&Image);
        neuron temp_n(th, Info[0][2],Info[0][3], Im);
        //cout<<"rec n = "<<Info[0][2]<<" rec m = "<<Info[0][3]<<" size ="<<Im.size()<<endl;
        //cout<<"temp_w = "<<temp_n.count_weights()<<endl;
        list_layers[0][i] = temp_n;
        list_layers[0][i].activate();
        list_layers[0][i].get_exit();
    }
    vector<maps*> temp_feachure_maps;

    for (int i = 1; i < n; i++) // создаем остальные слои
    {
        //cout<<"layer"<<i<<endl;
        list_layers[i].resize(Info[i][0]);
        temp_feachure_maps.resize(list_layers[i-1].size());


        for (int j = 0; j < list_layers[i-1].size(); j++)
        {
            temp_feachure_maps[j] = list_layers[i-1][j].get_exit();
        }

        for (int j = 0; j < list_layers[i].size(); j++)
        {
            neuron temp_n(th, Info[i][2],Info[i][3], temp_feachure_maps);
            list_layers[i][j] = temp_n;
            list_layers[i][j].activate();
            list_layers[i][j].get_exit();
        }

    }
    ///****создаем место, где будем хранить погрешности для весов и выходов нейронов******///

    d_outs.resize(Info.size());
    for (int i = 0; i < d_outs.size(); i++)
    {
        d_outs[i].resize(Info[i][0]);
        for(int j = 0; j < d_outs[i].size(); j++)
        {
            d_outs[i][j].map_resize(list_layers[i][j].feature_map_n(),list_layers[i][j].feature_map_m());
        }
    }
    d_weights.resize(Info.size());
    for (int i = 0; i < d_weights.size(); i++)
    {
        d_weights[i].resize(d_outs[i].size());
        for(int j = 0; j < d_outs[i].size(); j++)
        {
            d_weights[i][j].resize(list_layers[i][j].count_weights());
        }
    }

}

void net::back_propogation(vector<double> target, maps Image)
{
    double speed = 0.2;
    vector<double> a(list_layers[list_layers.size()-1].size());//здесь храним выходы последнего слоя

    ///*****прогон картинки****////
    result(Image);
    //info();

    ///****ошибка на последнем слое****///
    for (int i = 0; i < d_outs[d_outs.size()-1].size(); i++)
    {
        a[i] = list_layers[list_layers.size()-1][i].get_EXIT();
        d_outs[d_outs.size()-1][i][0][0] = a[i] - target[i];
        //cout<<"a[i] - target[i]"<<d_outs[d_outs.size()-1][i][0][0]<<endl;
        int n = list_layers[list_layers.size()-1][i].inputs_n(0);
        int m = list_layers[list_layers.size()-1][i].inputs_m(0);
        //cout<<"\nfor last layer neuron number "<<i<<endl;
        for (int j = 0; j < list_layers[list_layers.size()-1][i].count_weights(); j++)
        {
            int v,k,l;
            v = j / (n*m);
            l = (j - n*m*v) / n;
            k = j - n*m*v - l*n;
            maps temp = *list_layers[list_layers.size()-1][i].get_inputs()[v];
            double t = temp[l][k];
            d_weights[d_weights.size()-1][i][j] = speed*d_outs[d_outs.size()-1][i][0][0] * d_th(list_layers[list_layers.size()-1][i].Sigma[0][0]) * t ;
            //cout<<"\nw "<<list_layers.size()-1<<" i "<<i<<" j "<<j<<" = "<<t<<endl;
            //list_layers[list_layers.size()-1][i].change_weights(j,d_weights[d_weights.size()-1][i][j]);
        }
    }
    ///****ошибка на остальных слоях****///
    for (int nl = list_layers.size() - 2; nl > -1; nl--)//nl = layer's number
    {
        // создаем массивы для весов и ячеек связывающие карту след. слоя с предыдущим
        vector<vector<vector<int> > > mas_weights, mas_cell;
        mas_weights.resize(d_outs[nl][0].n);
        mas_cell.resize(d_outs[nl][0].n);
        for (int i = 0; i < d_outs[nl][0].n; i++)
        {
            mas_weights[i].resize(d_outs[nl][0].m);
            mas_cell[i].resize(d_outs[nl][0].m);
        }
        //cout<<"\nok nl = "<<nl<<endl;
        int n_next = d_outs[nl+1][0].n, m_next = d_outs[nl+1][0].m;//sizes of next maps
        //cout<<"n_next = "<<n_next<<"  m_next = "<<m_next<<endl;

        for (int i = 0; i < n_next; i++)
        {
            int sh_n = 1, sh_m = 1;
            int Begin_i = i * sh_n;
            for(int j = 0; j < m_next; j++)
            {
                int Begin_j = j * sh_m;
                for (int n = Begin_i; n < Begin_i+list_layers[nl+1][0].get_rec_n(); n++)
                    for (int m = Begin_j; m < Begin_j+list_layers[nl+1][0].get_rec_m(); m++)
                    {
                        mas_cell[n][m].push_back(i*m_next+j);
                        mas_weights[n][m].push_back((n-Begin_i)*(list_layers[nl+1][0].get_rec_m())+(m-Begin_j));
                        //if (nl == 2) cout<<"n ="<<n<<" m ="<<m<<" cell ="<<(i*m_next+j)<< "wieght = "<<((n-Begin_i)*(list_layers[nl+1][0].get_rec_m())+(m-Begin_j))<<endl;

                    }
            }
        }
        //cout<<"\nok1 nl = "<<nl<<endl;

        // получаем распределение ошибки на картах нейронов
        for (int i = 0; i < list_layers[nl].size(); i++)//номер карты нейронов
        {
            for (int k = 0; k < d_outs[nl][i].n; k++)
            {
                for (int m = 0; m < d_outs[nl][i].m; m++)
                {
                    d_outs[nl][i][k][m] = 0;
                    for (int v = 0; v < list_layers[nl+1].size(); v++)//количество карт в след. слое
                    {
                        //cout<<"\nlist_layers[nl+1].size()"<<list_layers[nl+1].size()<<endl;
                        //if (nl ==1 )cout<<"mas_cell[k][m].size() ="<<mas_cell[k][m].size()<<endl;
                        for (int p = 0; p < mas_cell[k][m].size(); p++)
                        {
                            //if (nl ==1 )cout<<"nl = "<<nl<<" i = "<<i<<" k = "<<k<<" m = "<<m <<" weightt ="<<mas_weights[k][m][p]+i*list_layers[nl+1][v].get_rec_n()*list_layers[nl+1][v].get_rec_m() <<endl;
                            double temp_weight = list_layers[nl+1][v].get_weight(mas_weights[k][m][p] + i*list_layers[nl+1][v].get_rec_n()*list_layers[nl+1][v].get_rec_m());
                            //cout<<"temp_weight = "<< temp_weight<<endl;
                            int i_ = mas_cell[k][m][p]/d_outs[nl+1][v].n, j_ = mas_cell[k][m][p]- d_outs[nl+1][v].m*i_;
                            d_outs[nl][i][k][m]+= d_outs[nl+1][v][i_][j_]*d_th(list_layers[nl+1][v].Sigma[i_][j_])*temp_weight;
                            //cout<<"nl = "<<nl<<"i = "<<i<<"k = "<<k<<" m = "<<m <<" outs ="<<d_outs[nl][i][k][m]<<endl;
                        }
                    }

                   //if(nl==1) cout<<"nl = "<<nl<<"i = "<<i<<"k = "<<k<<" m = "<<m <<" outs ="<<d_outs[nl][i][k][m]<<endl;
                }
            }

            //получаем распределение ошибки на весах //
            int n = list_layers[nl][i].inputs_n(0);
            int m = list_layers[nl][i].inputs_m(0);
            //cout<<"\n n = "<<n<<" m = "<<m<<endl;
            for (int j = 0; j < d_weights[nl][i].size(); j++)
            {
                d_weights[nl][i][j] = 0;
                vector<maps*> ttemp = list_layers[nl][i].get_inputs();
                double p = 0;
                for (int i_ = 0; i_ < d_outs[nl][i].n; i_++)
                    for (int j_ = 0; j_ < d_outs[nl][i][i_].size(); j_++)
                    {
                        int v,k,l;
                        v = j / (n*m);
                        l = (j - n*m*v) / n;
                        k = j - n*m*v - l*n;
                        maps temp = *list_layers[nl][i].get_inputs()[v];
                        double t = temp[l][k];
                        d_weights[nl][i][j] += speed*d_outs[nl][i][i_][j_] * d_th(list_layers[nl][i].Sigma[i_][j_]) * t ;
                    }
                list_layers[nl][i].change_weights(j,d_weights[nl][i][j]);
            }
            for (int k = 0; k < d_outs[nl][i].n; k++)
            {
                for (int m = 0; m < d_outs[nl][i].m; m++)
                {

                }
            }

        }
    }
    for(int i = 0; i < list_layers[list_layers.size()-1].size(); i++)
    {
        for (int j = 0; j < list_layers[list_layers.size()-1][i].count_weights(); j++)
            list_layers[list_layers.size()-1][i].change_weights(j,d_weights[d_weights.size()-1][i][j]);
    }
    //info();
}

void net::result(maps IMAGE)
{
    vector<maps*> Im;
    Image = IMAGE;
    Im.push_back(&Image);
    for (int i = 0; i < list_layers[0].size(); i++) // первый слой
    {
        list_layers[0][i].change_inputs(Im);
        list_layers[0][i].convolution();
    }
    vector<maps*> temp_feachure_maps;
    for (int i = 1; i < n; i++) // остальные слои
    {
        temp_feachure_maps.resize(list_layers[i-1].size());

        for (int j = 0; j < list_layers[i-1].size(); j++)
        {
            temp_feachure_maps[j] = list_layers[i-1][j].get_exit();
        }

        for (int j = 0; j < list_layers[i].size(); j++)
        {
            list_layers[i][j].change_inputs(temp_feachure_maps);
            list_layers[i][j].convolution();
            list_layers[i][j].get_exit();
        }
    }
    //info();
    cout<<"\n result"<<endl;
    for (int i = 0; i < list_layers[list_layers.size()-1].size(); i++)
        cout<<"\nY "<< i <<" = "<<list_layers[list_layers.size()-1][i].get_EXIT();
}

void net::info()
{
    cout<<"\n NET Info"<<endl;
    cout<<"Count of layers : "<<list_layers.size()<<endl;
    for (int i = 0; i < list_layers.size(); i++)
        {
            cout<<"layer "<<i<<" : "<<endl;
            cout<<"Count of neurons : "<<list_layers[i].size()<<endl;
            cout<<"Inputs : "<<endl;
            for (int j = 0; j < list_layers[i].size(); j++)
                {
                    list_layers[i][j].info_inputs();
                    cout<<endl;
                }

            }
    /*cout<<"\n Result :"<<endl;
    for (int i = 0; i < list_layers[list_layers.size()-1].size(); i++)
        cout<<"\nY "<< i <<" = "<<list_layers[list_layers.size()-1][i].get_EXIT();*/
    /*cout<<"\nFeture_map"<<endl;
    for (int i = 0; i < d.size(); i++)
    {
        cout<<"layer : "<<i<<endl;
        for (int j = 0; j < d_outs[i].size(); j++)
        {
            cout<<"f_map : "<<j<<endl;
            for(int q = 0; q < d_outs[i][j].n; q++)
            {
                for(int w = 0; w < d_outs[i][j].m; w++)
                    cout<<d_outs[i][j][q][w]<<"  ";
                cout<<endl;
            }
        }
        cout<<endl;
    }*/
    cout<<"\nWeights"<<endl;
    for (int i = 0; i < list_layers.size(); i++)
    {
        cout<<"layer : "<<i<<endl;
        for (int j = 0; j < list_layers[i].size(); j++)
        {
            cout<<"neuron : "<<j<<endl;
            for(int q = 0; q < list_layers[i][j].count_weights(); q++)
            {
                    cout<<list_layers[i][j].get_weight(q)<<"  ";
            }
            cout<<endl;
        }
        cout<<endl;
    }
}
void net::back_prop_info()
{
    cout<<"\nd_outs"<<endl;
    for (int i = 0; i < d_outs.size(); i++)
    {
        cout<<"layer : "<<i<<endl;
        for (int j = 0; j < d_outs[i].size(); j++)
        {
            cout<<"f_map : "<<j<<endl;
            for(int q = 0; q < d_outs[i][j].n; q++)
            {
                for(int w = 0; w < d_outs[i][j].m; w++)
                    cout<<d_outs[i][j][q][w]<<"  ";
                cout<<endl;
            }
        }
        cout<<endl;
    }
    cout<<"\nd_weights"<<endl;
    for (int i = 0; i < d_weights.size(); i++)
    {
        cout<<"layer : "<<i<<endl;
        for (int j = 0; j < d_weights[i].size(); j++)
        {
            cout<<"neuron : "<<j<<endl;
            for(int q = 0; q < d_weights[i][j].size(); q++)
            {
                    cout<<d_weights[i][j][q]<<"  ";
            }
            cout<<endl;
        }
        cout<<endl;
    }
}
