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
    for (int nl = list_layers.size() - 2; nl > -1; nl--)//cl = layer's number
    {
        // получаем распределение ошибки на картах нейронов
        for (int i = 0; i < list_layers[nl].size(); i++)//номер карты нейронов
        {
            int m_b_i[d_outs[nl][i].n], m_e_i[d_outs[nl][i].n];//чтобы посмотреть к каким нейронам относится данный выход
            int m_b_j[d_outs[nl][i].m], m_e_j[d_outs[nl][i].m];
            int temp_rec_n = list_layers[nl+1][0].get_rec_n() , temp_rec_m = list_layers[nl+1][0].get_rec_m();
            //cout<<"\n temp_rec_n = "<<temp_rec_n<<endl;
            for (int j = 0; j < d_outs[nl][i].n; j++) if (j < temp_rec_n) m_b_i[j] = 0; else m_b_i[j] = m_b_i[j-1] + 1;
            for (int j = 0; j < d_outs[nl][i].m; j++) if (j < temp_rec_m) m_b_j[j] = 0; else m_b_j[j] = m_b_j[j-1] + 1;
            for (int j = 0; j < (d_outs[nl][i].n + 1)/2; j++)
                    if (j < temp_rec_n) m_e_i[0 + j] = m_e_i[d_outs[nl][i].n - 1 - j] = j+1;
                    else m_e_i[0 + j] = m_e_i[d_outs[nl][i].n - 1 - j] = temp_rec_n;
            for (int j = 0; j < (d_outs[nl][i].m + 1)/2; j++)
                    if (j < temp_rec_m) m_e_j[0 + j] = m_e_j[d_outs[nl][i].m - 1 - j] = j+1;
                    else m_e_j[0 + j] = m_e_j[d_outs[nl][i].m - 1 - j] = temp_rec_m;
            //for (int j = 0; j < temp_rec_n; j++) cout<<"\n begin = "<<m_b_i[j]<<" end = "<<m_e_i[j]<<endl;
            for (int k = 0; k < d_outs[nl][i].n; k++)
            {
                for (int m = 0; m < d_outs[nl][i].m; m++)
                {
                    d_outs[nl][i][k][m] = 0;
                    for (int v = 0; v < list_layers[nl+1].size(); v++)
                        for (int i_ = 0; i_ < m_b_i[k]+m_e_i[k]; i_++)
                            for (int j_ = 0; j_ < m_b_j[m]+m_e_j[m]; j_++)
                        {
                            int temp_weight = i*d_outs[nl][i].n*d_outs[nl][i].m+k*d_outs[nl][i].n+m;
                            if(nl == 1)cout<<"\nnl = "<<nl<<" neqron = "<<i<<" k = "<<k<<" m = "<<m<<" temp_weight = "<<temp_weight<<"  "<<list_layers[nl+1][v].get_weight(temp_weight)<<endl;
                            d_outs[nl][i][k][m]+= d_outs[nl+1][v][i_][j_]*d_th(list_layers[nl+1][v].Sigma[i_][j_])*list_layers[nl+1][v].get_weight(temp_weight);
                            if(abs(d_outs[nl][i][k][m]) >= 1) d_outs[nl][i][k][m] = 1;
                            //cout<<"\nweight = "<<list_layers[nl+1][v].get_weight(temp_weight)<<" vkm = "<<temp_weight<<endl;
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
                        //cout<<"\n v = "<<v<<" l = "<<l<<" k = "<<k<<endl;
                        maps temp = *list_layers[nl][i].get_inputs()[v];

                        //cout<<"ttemp = "<<ttemp.size()<<endl;
                        /*maps *temp = ttemp[v];
                        cout<<temp->map[i_][j_]<<" ";*/
                        //cout<<"temp = "<<temp->map[0][1]<<endl;
                        double t = temp[l][k];
                        d_weights[nl][i][j] += speed*d_outs[nl][i][i_][j_] * d_th(list_layers[nl][i].Sigma[i_][j_]) * t ;
                        //p += d_outs[nl][i][i_][j_];
                    }
                list_layers[nl][i].change_weights(j,d_weights[nl][i][j]);
                //cout<<"\nw "<<nl<<" i "<<i<<" j "<<j<<" = "<<p<<endl;
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
