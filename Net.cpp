#include "Headers.h"
#include <cmath>
using namespace std;
using namespace activation_functions;

net::net(maps & IMAGE, vector<vector<int> > Info)
{

    n = Info.size();// количество слоев

    list_layers.resize(n);
    list_layers[0].resize(Info[0][0]);// создаем первый слой
    vector<maps*> Im;
    Image = IMAGE;
    Im.push_back(&Image);
    average_weight = 0;



    //cout<<"\n zero layer"<<endl;
    list_layers[0] = layer(list_layers[0].size(), th, Info[0][1],Info[0][2], Im, Info[0][3], Info[0][4], Info[0][5]);

    //list_layers[0].info();
    vector<maps*> temp_feachure_maps;

    for (int i = 1; i < n; i++) // создаем остальные слои
    {
        //cout<<"Flag1"<<endl;
       // temp_feachure_maps.resize(list_layers[i-1].size());
        temp_feachure_maps = list_layers[i-1].Get_Outs();
        //cout<<"Flag2"<<endl;
        layer Temp(Info[i][0], th, Info[i][1], Info[i][2], temp_feachure_maps, Info[i][3], Info[i][4], Info[i][5]);
        //cout<<Info[i][5];
        list_layers[i] = Temp;

    }
    exit.resize(list_layers[list_layers.size() - 1].size());
    preparation_backprop();
}
void net::preparation_backprop()
{
    ///****создаем место, где будем хранить погрешности для весов и выходов нейронов******///

    d_outs.resize(list_layers.size());
    zero_d_weights.resize(list_layers.size());
    d_reduce_outs.resize(list_layers.size());
    for (int i = 0; i < d_outs.size(); i++)
    {
        d_outs[i].resize(list_layers[i].size()); // neurons
        if (list_layers[i].is_reduce()) d_reduce_outs[i].resize(list_layers[i].size());//neurons
        else d_reduce_outs[i].resize(0);
        zero_d_weights[i].resize(list_layers[i].size());//w0
        for(int j = 0; j < d_outs[i].size(); j++)
        {
            zero_d_weights[i][j] = 0;
            if (list_layers[i].is_reduce()) {
                    d_outs[i][j].map_resize(list_layers[i][j].final_f_map_n(), list_layers[i][j].final_f_map_m());
                    d_reduce_outs[i][j].map_resize(list_layers[i][j].feature_map_n(),list_layers[i][j].feature_map_m());
                    //cout<<"layer "<<i<<" size of n = "<<list_layers[i][j].final_f_map_n()<<" size of m = "<<list_layers[i][j].final_f_map_m();
            }
            else
            {
                d_outs[i][j].map_resize(list_layers[i][j].feature_map_n(),list_layers[i][j].feature_map_m());
            }


        }
    }
    d_weights.resize(list_layers.size());
    for (int i = 0; i < d_weights.size(); i++)
    {
        d_weights[i].resize(d_outs[i].size());
        for(int j = 0; j < d_outs[i].size(); j++)
        {
            d_weights[i][j].resize(list_layers[i][j].count_weights());
        }
    }
}
void net::back_propogation(vector<double> target, maps Image, double Speed)
{
    double speed = Speed;
    vector<double> a(list_layers[list_layers.size()-1].size());//здесь храним выходы последнего слоя
    ///*****прогон картинки****////
    result(Image);
    ///****ошибка на последнем слое****///
    for (int i = 0; i < d_outs[d_outs.size()-1].size(); i++)
    {
        a[i] = list_layers[list_layers.size()-1][i].get_EXIT();
        d_outs[d_outs.size()-1][i][0][0] = a[i] - target[i];
        int n = list_layers[list_layers.size()-1][i].inputs_n(0);
        int m = list_layers[list_layers.size()-1][i].inputs_m(0);
        for (int j = 0; j < list_layers[list_layers.size()-1][i].count_weights(); j++)
        {
            int v,k,l;
            v = j / (n*m);
            l = (j - n*m*v) / n;
            k = j - n*m*v - l*n;
            maps temp = *list_layers[list_layers.size()-1][i].get_inputs()[v];
            double t = temp[l][k];
            d_weights[d_weights.size()-1][i][j] = d_outs[d_outs.size()-1][i][0][0] * d_th(list_layers[list_layers.size()-1][i].Sigma[0][0]) * t ;
            zero_d_weights[d_outs.size()-1][i] = d_outs[d_outs.size()-1][i][0][0] * d_th(list_layers[list_layers.size()-1][i].Sigma[0][0]);
        }
    }

    ///****ошибка на остальных слоях****///
    for (int nl = list_layers.size() - 2; nl > -1; nl--)//nl = layer's number
    {
        // создаем массивы для весов и ячеек связывающие карту след. слоя с предыдущим
        vector<vector<vector<int> > > mas_weights, mas_cell;
        mas_weights.resize(d_outs[nl][0].n);
        mas_cell.resize(d_outs[nl][0].n);
        int n_next,m_next;
        for (int i = 0; i < d_outs[nl][0].n; i++)
            {
                mas_weights[i].resize(d_outs[nl][0].m);
                mas_cell[i].resize(d_outs[nl][0].m);
            }

        if (list_layers[nl+1].is_reduce())
            {
                n_next = d_reduce_outs[nl+1][0].n; m_next = d_reduce_outs[nl+1][0].m;//sizes of next maps
            }
        else
            {
                n_next = d_outs[nl+1][0].n; m_next = d_outs[nl+1][0].m;//sizes of next maps
            }

        for (int i = 0; i < n_next; i++)
        {
            int sh_n = list_layers[nl+1][0].get_shift_n(), sh_m = list_layers[nl+1][0].get_shift_m();
            int Begin_i = i * sh_n;
            for(int j = 0; j < m_next; j++)
            {
                int Begin_j = j * sh_m;
                for (int n = Begin_i; n < Begin_i+list_layers[nl+1][0].get_rec_n(); n++)
                    for (int m = Begin_j; m < Begin_j+list_layers[nl+1][0].get_rec_m(); m++)
                    {
                        mas_cell[n][m].push_back(i*m_next+j);
                        mas_weights[n][m].push_back((n-Begin_i)*(list_layers[nl+1][0].get_rec_m())+(m-Begin_j));
                    }
            }
        }

        // получаем распределение ошибки на картах нейронов
        for (int i = 0; i < list_layers[nl].size(); i++)//номер карты нейронов
        {
            for (int k = 0; k < d_outs[nl][i].n; k++)
            {
                for (int m = 0; m < d_outs[nl][i].m; m++)
                {
                    //d_outs[nl][i][k][m] = 0;
                    for (int v = 0; v < list_layers[nl+1].size(); v++)//количество карт в след. слое
                    {
                        for (int p = 0; p < mas_cell[k][m].size(); p++)
                        {
                            double temp_weight = list_layers[nl+1][v].get_weight(mas_weights[k][m][p] + i*list_layers[nl+1][v].get_rec_n()*list_layers[nl+1][v].get_rec_m());
                            int i_=0,j_=0;

                            if (list_layers[nl].is_reduce())
                            {
                                if (list_layers[nl+1].is_reduce())
                                {
                                    i_ = mas_cell[k][m][p]/d_reduce_outs[nl+1][v].n; j_ = mas_cell[k][m][p]- d_reduce_outs[nl+1][v].m*i_;
                                    d_outs[nl][i][k][m]+= d_reduce_outs[nl+1][v][i_][j_]*d_th(list_layers[nl+1][v].Sigma[i_][j_])*temp_weight;
                                }
                                else
                                {
                                    i_ = mas_cell[k][m][p]/d_outs[nl+1][v].n; j_ = mas_cell[k][m][p]- d_outs[nl+1][v].m*i_;
                                    d_outs[nl][i][k][m]+= d_outs[nl+1][v][i_][j_]*d_th(list_layers[nl+1][v].Sigma[i_][j_])*temp_weight;
                                }

                            }
                            else
                            {
                                if (list_layers[nl+1].is_reduce())
                                {
                                    i_ = mas_cell[k][m][p]/d_reduce_outs[nl+1][v].n; j_ = mas_cell[k][m][p]- d_reduce_outs[nl+1][v].m*i_;
                                    d_outs[nl][i][k][m]+= d_reduce_outs[nl+1][v][i_][j_]*d_th(list_layers[nl+1][v].Sigma[i_][j_])*temp_weight;
                                }
                                else
                                {
                                    i_ = mas_cell[k][m][p]/d_outs[nl+1][v].n; j_ = mas_cell[k][m][p]- d_outs[nl+1][v].m*i_;
                                    d_outs[nl][i][k][m]+= d_outs[nl+1][v][i_][j_]*d_th(list_layers[nl+1][v].Sigma[i_][j_])*temp_weight;
                                }
                            }
                        }
                    }
                }
            }

            if (list_layers[nl].is_reduce())// if reduce we get mistakes on d_reduce_feature map
            {
                int temp = list_layers[nl][0].kernel();
                for (int k = 0; k < d_outs[nl][i].n; k++)
                {
                    for (int m = 0; m < d_outs[nl][i].m; m++)
                    {
                        for (int u = temp * k; u < temp * k + temp; u++)
                            for (int w = temp * m; w < temp * m + temp; w++)
                                if (list_layers[nl+1].is_reduce())
                                {
                                    d_reduce_outs[nl][i][u][w] = 1/(temp*temp) * d_reduce_outs[nl][i][k][m];
                                }
                                else
                                {
                                    d_reduce_outs[nl][i][u][w] = 1/(temp*temp) * d_outs[nl][i][k][m];
                                }
                    }
                }

            }
            //получаем распределение ошибки на весах //
            int n = list_layers[nl][i].inputs_n(0); //sizes of image to enter
            int m = list_layers[nl][i].inputs_m(0); //
            for (int j = 0; j < d_weights[nl][i].size(); j++)
                {
                    d_weights[nl][i][j] = 0;
                }
            // mas_weights and mas_cell for prev. layer
            vector<vector<vector<int> > > new_mas_weights, new_mas_cell;

            new_mas_weights.resize(n);
            new_mas_cell.resize(n);
            for (int i = 0; i < n; i++)
                {
                    new_mas_weights[i].resize(m);
                    new_mas_cell[i].resize(m);
                }
            //********************************sizes of this maps
            if (list_layers[nl].is_reduce())
                {
                    int n_next = d_reduce_outs[nl][0].n, m_next = d_reduce_outs[nl][0].m;
                }
            else
                {
                    int n_next = d_outs[nl][0].n, m_next = d_outs[nl][0].m;
                }

            for (int i = 0; i < n_next; i++)
            {

                int sh_n = list_layers[nl][0].get_shift_n(), sh_m = list_layers[nl][0].get_shift_m(); //// change
                int Begin_i = i * sh_n;
                for(int j = 0; j < m_next; j++)
                {
                    int Begin_j = j * sh_m;
                    for (int n = Begin_i; n < Begin_i+list_layers[nl][0].get_rec_n(); n++)
                        for (int m = Begin_j; m < Begin_j+list_layers[nl][0].get_rec_m(); m++)
                        {

                            new_mas_cell[n][m].push_back(i*m_next+j);
                            new_mas_weights[n][m].push_back((n-Begin_i)*(list_layers[nl][0].get_rec_m())+(m-Begin_j));
                        }
                }
            }
            //zero_d_weights[nl][i] = 0;
            for (int i_ = 0; i_ < n_next; i_++)
                for(int j_ = 0; j_ < m_next; j_++)
                    if (list_layers[nl].is_reduce())
                        {
                            zero_d_weights[nl][i]+= d_reduce_outs[nl][i][i_][j_]*d_th(list_layers[nl][i].Sigma[i_][j_]);

                        }
                    else
                        {
                            zero_d_weights[nl][i]+= d_outs[nl][i][i_][j_]*d_th(list_layers[nl][i].Sigma[i_][j_]);
                        }


            for (int count_prev_n_l = 0; count_prev_n_l < list_layers[nl][i].count_of_maps(); count_prev_n_l++)
            {
                for (int k = 0; k < list_layers[nl][i].inputs_n(0); k++)// |
                {
                    for (int t = 0; t < list_layers[nl][i].inputs_m(0); t++)// -
                    {
                        for (int p = 0; p < new_mas_cell[k][t].size(); p++)// count in cell
                        {
                            if (list_layers[nl].is_reduce())
                            {
                                int i_ = new_mas_cell[k][t][p]/d_reduce_outs[nl][i].m, j_ = new_mas_cell[k][t][p] - i_*d_reduce_outs[nl][i].m;
                                maps temp = *list_layers[nl][i].get_inputs()[count_prev_n_l];
                                double temp_inp = temp[k][t];
                                d_weights[nl][i][new_mas_weights[k][t][p] + count_prev_n_l*list_layers[nl][i].get_rec_n()*list_layers[nl][i].get_rec_m()] += d_reduce_outs[nl][i][i_][j_]*d_th(list_layers[nl][i].Sigma[i_][j_])*temp_inp;
                            }
                            else
                            {
                                int i_ = new_mas_cell[k][t][p]/d_outs[nl][i].m, j_ = new_mas_cell[k][t][p] - i_*d_outs[nl][i].m;
                                maps temp = *list_layers[nl][i].get_inputs()[count_prev_n_l];
                                double temp_inp = temp[k][t];
                                d_weights[nl][i][new_mas_weights[k][t][p] + count_prev_n_l*list_layers[nl][i].get_rec_n()*list_layers[nl][i].get_rec_m()] += d_outs[nl][i][i_][j_]*d_th(list_layers[nl][i].Sigma[i_][j_])*temp_inp;
                            }
                        }
                    }
                }
            }
        }
    }

}
void net::update_weights(double step)
{
    // change weights
    int count_w_in_n = 0;
    for(int i = 0; i < d_weights.size(); i++)
    {
        for (int j = 0; j < d_weights[i].size(); j++)
        {
            count_w_in_n += 1;
            //speed +=0.01;
            list_layers[i][j].change_zero_weights(step*zero_d_weights[i][j]);
            average_weight += sqrt(step * zero_d_weights[i][j] * step * zero_d_weights[i][j]);
            for(int q = 0; q < list_layers[i][j].count_weights(); q++)
            {
                list_layers[i][j].change_weights(q, step*d_weights[i][j][q]);
                average_weight += sqrt(step * d_weights[i][j][q] * step * d_weights[i][j][q]);
                count_w_in_n += 1;
            }
        }
    }
     average_weight /= count_w_in_n;////////////////!!!!!!!!!!!!!!!
}
void net::back_propogation_online(vector<double> target, maps Image, double Speed)
{
    zeroize_diff();
    back_propogation(target, Image, Speed);
    update_weights(Speed);

}
void net::back_propogation_offline(vector<vector<double> > &target, vector<maps> &Image, double Speed, bool isForGen)
{
    zeroize_diff();
    vector<double> a(list_layers[list_layers.size()-1].size());//здесь храним выходы последнего слоя

    for (int pocket_size = 0; pocket_size < target.size(); pocket_size++)
    {
        ///*****прогон картинки****////
        result(Image[pocket_size]);
        ///****ошибка на последнем слое****///
        for (int i = 0; i < d_outs[d_outs.size()-1].size(); i++)
        {
            a[i] = list_layers[list_layers.size()-1][i].get_EXIT();
            d_outs[d_outs.size()-1][i][0][0] += (a[i] - target[pocket_size][i])/target.size();
        }

    }
    for (int i = 0; i < d_outs[d_outs.size()-1].size(); i++)
    {
        int n = list_layers[list_layers.size()-1][i].inputs_n(0);
        int m = list_layers[list_layers.size()-1][i].inputs_m(0);
        for (int j = 0; j < list_layers[list_layers.size()-1][i].count_weights(); j++)
        {
            int v,k,l;
            v = j / (n*m);
            l = (j - n*m*v) / n;
            k = j - n*m*v - l*n;
            maps temp = *list_layers[list_layers.size()-1][i].get_inputs()[v];
            double t = temp[l][k];
            d_weights[d_weights.size()-1][i][j] = d_outs[d_outs.size()-1][i][0][0] * d_th(list_layers[list_layers.size()-1][i].Sigma[0][0]) * t ;
            zero_d_weights[d_outs.size()-1][i] = d_outs[d_outs.size()-1][i][0][0] * d_th(list_layers[list_layers.size()-1][i].Sigma[0][0]);
        }
    }

    ///****ошибка на остальных слоях****///
    for (int nl = list_layers.size() - 2; nl > -1; nl--)//nl = layer's number
    {
        // создаем массивы для весов и ячеек связывающие карту след. слоя с предыдущим
        vector<vector<vector<int> > > mas_weights, mas_cell;
        mas_weights.resize(d_outs[nl][0].n);
        mas_cell.resize(d_outs[nl][0].n);
        int n_next,m_next;
        for (int i = 0; i < d_outs[nl][0].n; i++)
            {
                mas_weights[i].resize(d_outs[nl][0].m);
                mas_cell[i].resize(d_outs[nl][0].m);
            }

        if (list_layers[nl+1].is_reduce())
            {
                n_next = d_reduce_outs[nl+1][0].n; m_next = d_reduce_outs[nl+1][0].m;//sizes of next maps
            }
        else
            {
                n_next = d_outs[nl+1][0].n; m_next = d_outs[nl+1][0].m;//sizes of next maps
            }

        for (int i = 0; i < n_next; i++)
        {
            int sh_n = list_layers[nl+1][0].get_shift_n(), sh_m = list_layers[nl+1][0].get_shift_m();
            int Begin_i = i * sh_n;
            for(int j = 0; j < m_next; j++)
            {
                int Begin_j = j * sh_m;
                for (int n = Begin_i; n < Begin_i+list_layers[nl+1][0].get_rec_n(); n++)
                    for (int m = Begin_j; m < Begin_j+list_layers[nl+1][0].get_rec_m(); m++)
                    {
                        mas_cell[n][m].push_back(i*m_next+j);
                        mas_weights[n][m].push_back((n-Begin_i)*(list_layers[nl+1][0].get_rec_m())+(m-Begin_j));
                    }
            }
        }

        // получаем распределение ошибки на картах нейронов
        for (int i = 0; i < list_layers[nl].size(); i++)//номер карты нейронов
        {
            for (int k = 0; k < d_outs[nl][i].n; k++)
            {
                for (int m = 0; m < d_outs[nl][i].m; m++)
                {
                    //d_outs[nl][i][k][m] = 0;
                    for (int v = 0; v < list_layers[nl+1].size(); v++)//количество карт в след. слое
                    {
                        for (int p = 0; p < mas_cell[k][m].size(); p++)
                        {
                            double temp_weight = list_layers[nl+1][v].get_weight(mas_weights[k][m][p] + i*list_layers[nl+1][v].get_rec_n()*list_layers[nl+1][v].get_rec_m());
                            int i_=0,j_=0;

                            if (list_layers[nl].is_reduce())
                            {
                                if (list_layers[nl+1].is_reduce())
                                {
                                    i_ = mas_cell[k][m][p]/d_reduce_outs[nl+1][v].n; j_ = mas_cell[k][m][p]- d_reduce_outs[nl+1][v].m*i_;
                                    d_outs[nl][i][k][m]+= d_reduce_outs[nl+1][v][i_][j_]*d_th(list_layers[nl+1][v].Sigma[i_][j_])*temp_weight;
                                }
                                else
                                {
                                    i_ = mas_cell[k][m][p]/d_outs[nl+1][v].n; j_ = mas_cell[k][m][p]- d_outs[nl+1][v].m*i_;
                                    d_outs[nl][i][k][m]+= d_outs[nl+1][v][i_][j_]*d_th(list_layers[nl+1][v].Sigma[i_][j_])*temp_weight;
                                }

                            }
                            else
                            {
                                if (list_layers[nl+1].is_reduce())
                                {
                                    i_ = mas_cell[k][m][p]/d_reduce_outs[nl+1][v].n; j_ = mas_cell[k][m][p]- d_reduce_outs[nl+1][v].m*i_;
                                    d_outs[nl][i][k][m]+= d_reduce_outs[nl+1][v][i_][j_]*d_th(list_layers[nl+1][v].Sigma[i_][j_])*temp_weight;
                                }
                                else
                                {
                                    i_ = mas_cell[k][m][p]/d_outs[nl+1][v].n; j_ = mas_cell[k][m][p]- d_outs[nl+1][v].m*i_;
                                    d_outs[nl][i][k][m]+= d_outs[nl+1][v][i_][j_]*d_th(list_layers[nl+1][v].Sigma[i_][j_])*temp_weight;
                                }
                            }
                        }
                    }
                }
            }

            if (list_layers[nl].is_reduce())// if reduce we get mistakes on d_reduce_feature map
            {
                int temp = list_layers[nl][0].kernel();
                for (int k = 0; k < d_outs[nl][i].n; k++)
                {
                    for (int m = 0; m < d_outs[nl][i].m; m++)
                    {
                        for (int u = temp * k; u < temp * k + temp; u++)
                            for (int w = temp * m; w < temp * m + temp; w++)
                                if (list_layers[nl+1].is_reduce())
                                {
                                    d_reduce_outs[nl][i][u][w] = 1/(temp*temp) * d_reduce_outs[nl][i][k][m];
                                }
                                else
                                {
                                    d_reduce_outs[nl][i][u][w] = 1/(temp*temp) * d_outs[nl][i][k][m];
                                }
                    }
                }

            }
            //получаем распределение ошибки на весах //
            int n = list_layers[nl][i].inputs_n(0); //sizes of image to enter
            int m = list_layers[nl][i].inputs_m(0); //
            for (int j = 0; j < d_weights[nl][i].size(); j++)
                {
                    d_weights[nl][i][j] = 0;
                }
            // mas_weights and mas_cell for prev. layer
            vector<vector<vector<int> > > new_mas_weights, new_mas_cell;

            new_mas_weights.resize(n);
            new_mas_cell.resize(n);
            for (int i = 0; i < n; i++)
                {
                    new_mas_weights[i].resize(m);
                    new_mas_cell[i].resize(m);
                }
            //********************************sizes of this maps
            if (list_layers[nl].is_reduce())
                {
                    int n_next = d_reduce_outs[nl][0].n, m_next = d_reduce_outs[nl][0].m;
                }
            else
                {
                    int n_next = d_outs[nl][0].n, m_next = d_outs[nl][0].m;
                }

            for (int i = 0; i < n_next; i++)
            {

                int sh_n = list_layers[nl][0].get_shift_n(), sh_m = list_layers[nl][0].get_shift_m(); //// change
                int Begin_i = i * sh_n;
                for(int j = 0; j < m_next; j++)
                {
                    int Begin_j = j * sh_m;
                    for (int n = Begin_i; n < Begin_i+list_layers[nl][0].get_rec_n(); n++)
                        for (int m = Begin_j; m < Begin_j+list_layers[nl][0].get_rec_m(); m++)
                        {

                            new_mas_cell[n][m].push_back(i*m_next+j);
                            new_mas_weights[n][m].push_back((n-Begin_i)*(list_layers[nl][0].get_rec_m())+(m-Begin_j));
                        }
                }
            }
            //zero_d_weights[nl][i] = 0;
            for (int i_ = 0; i_ < n_next; i_++)
                for(int j_ = 0; j_ < m_next; j_++)
                    if (list_layers[nl].is_reduce())
                        {
                            zero_d_weights[nl][i]+= d_reduce_outs[nl][i][i_][j_]*d_th(list_layers[nl][i].Sigma[i_][j_]);

                        }
                    else
                        {
                            zero_d_weights[nl][i]+= d_outs[nl][i][i_][j_]*d_th(list_layers[nl][i].Sigma[i_][j_]);
                        }


            for (int count_prev_n_l = 0; count_prev_n_l < list_layers[nl][i].count_of_maps(); count_prev_n_l++)
            {
                for (int k = 0; k < list_layers[nl][i].inputs_n(0); k++)// |
                {
                    for (int t = 0; t < list_layers[nl][i].inputs_m(0); t++)// -
                    {
                        for (int p = 0; p < new_mas_cell[k][t].size(); p++)// count in cell
                        {
                            if (list_layers[nl].is_reduce())
                            {
                                int i_ = new_mas_cell[k][t][p]/d_reduce_outs[nl][i].m, j_ = new_mas_cell[k][t][p] - i_*d_reduce_outs[nl][i].m;
                                maps temp = *list_layers[nl][i].get_inputs()[count_prev_n_l];
                                double temp_inp = temp[k][t];
                                d_weights[nl][i][new_mas_weights[k][t][p] + count_prev_n_l*list_layers[nl][i].get_rec_n()*list_layers[nl][i].get_rec_m()] += d_reduce_outs[nl][i][i_][j_]*d_th(list_layers[nl][i].Sigma[i_][j_])*temp_inp;
                            }
                            else
                            {
                                int i_ = new_mas_cell[k][t][p]/d_outs[nl][i].m, j_ = new_mas_cell[k][t][p] - i_*d_outs[nl][i].m;
                                maps temp = *list_layers[nl][i].get_inputs()[count_prev_n_l];
                                double temp_inp = temp[k][t];
                                d_weights[nl][i][new_mas_weights[k][t][p] + count_prev_n_l*list_layers[nl][i].get_rec_n()*list_layers[nl][i].get_rec_m()] += d_outs[nl][i][i_][j_]*d_th(list_layers[nl][i].Sigma[i_][j_])*temp_inp;
                            }
                        }
                    }
                }
            }
        }
    }
    //back_prop_info();
    if (!isForGen)update_weights(Speed);
}

int net::result(maps IMAGE, bool flag)
{
    vector<maps*> Im;
    Image = IMAGE;
    Im.push_back(&Image);
    for (int i = 0; i < list_layers[0].size(); i++) // первый слой
    {
        list_layers[0][i].change_inputs(Im);
        if (!list_layers[0][i].Act) list_layers[0][i].activate();
        list_layers[0][i].convolution();
    }
    vector<maps*> temp_feachure_maps;
    for (int i = 1; i < n; i++) // остальные слои
    {
        temp_feachure_maps.resize(list_layers[i-1].size());

        for (int j = 0; j < list_layers[i-1].size(); j++)
        {
            temp_feachure_maps[j] = list_layers[i-1][j].get_exit();
            //if (i==1)list_layers[i-1][j].info_feature();
        }

        for (int j = 0; j < list_layers[i].size(); j++)
        {
            list_layers[i][j].change_inputs(temp_feachure_maps);
            list_layers[i][j].convolution();
            list_layers[i][j].get_exit();
            //if (j==0) list_layers[i][j].info_inputs();
        }
    }
    vector<double> temp(list_layers[list_layers.size()-1].size());
    double max = temp[0];
    int pos = 0;
    for (int i = 0; i < list_layers[list_layers.size()-1].size(); i++)
        {

            //cout<<"\nY "<< i <<" = "<<list_layers[list_layers.size()-1][i].get_EXIT();
            temp[i] = list_layers[list_layers.size()-1][i].get_EXIT();
        }
        for (int i = 0; i < temp.size(); i++)
            if (max < temp[i]) { max = temp[i]; pos = i;}

    if(flag)
    {
        /*cout<<"\n result"<<endl;
        for (int i = 0; i < list_layers[0].size(); i++)
        {
            list_layers[0][i].info_inputs();
        }*/
        for (int i = 0; i < list_layers[list_layers.size()-1].size(); i++)
        {

            cout<<"\nY "<< i <<" = "<<list_layers[list_layers.size()-1][i].get_EXIT();
            //temp[i] = list_layers[list_layers.size()-1][i].get_EXIT();
        }
    }
    return pos;
}
void net::activate(maps IMAGE)
{
    vector<maps*> Im;
    Image = IMAGE;
    Im.push_back(&Image);
    for (int i = 0; i < list_layers[0].size(); i++) // первый слой
    {
        list_layers[0][i].change_inputs(Im);
        list_layers[0][i].activate();
        list_layers[0][i].convolution();
        list_layers[0][i].get_exit();
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
            list_layers[i][j].activate();
            list_layers[i][j].convolution();
            list_layers[i][j].get_exit();
        }
    }
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
            for (int j = 0; j < 1; j++)//list_layers[i].size()
                {
                    list_layers[i][j].info_inputs();
                    cout<<endl;

                }
            for (int j = 0; j < list_layers[i].size(); j++)
            {
                list_layers[i][j].info_feature();
                    cout<<endl;
            }

        }

    /*cout<<"\nWeights"<<endl;
    for (int i = 0; i < list_layers.size(); i++)
    {
        cout<<"layer : "<<i<<endl;
        for (int j = 0; j < list_layers[i].size(); j++)
        {
            cout<<"neuron : "<<j<<endl;
            cout<<"zero_weight = "<< list_layers[i][j].get_zero_weight()<<endl;
            for(int q = 0; q < list_layers[i][j].count_weights(); q++)
            {
                    cout<<list_layers[i][j].get_weight(q)<<"  ";
            }
            cout<<endl;
        }
        cout<<endl;
    }*/
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
void net::add_layer(int number, int count_neurons, int rec_n, int rec_m, int sh_n, int sh_m, int Ker)
{
    n ++;
}
void net::add_neuron(int position_in_layer, int layers_number)
{
    if ( layers_number == n ) exit.insert(exit.begin()+position_in_layer, 0);
}

void net::save_net(const char* filename)
{
    ofstream File;
    File.open(filename);

    File<<list_layers.size()<<endl;
    for (int i = 0; i < list_layers.size(); i++)
    {
        File<<list_layers[i].size()<<endl; //кол.нейроннов
        File<<list_layers[i][0].get_rec_n()<<" "<<list_layers[i][0].get_rec_m()<<endl;
        File<<list_layers[i][0].get_shift_n()<<" "<<list_layers[i][0].get_shift_m()<<endl;
        File<<list_layers[i][0].kernel()<<endl;
        File<<list_layers[i][0].inp_size<<" "<<list_layers[i][0].image_n<<" "<<list_layers[i][0].image_m<<endl;

        for (int j = 0; j < list_layers[i].size(); j++)
        {
            File<<list_layers[i][j].get_zero_weight()<<endl;
            for (int count_w = 0; count_w < list_layers[i][j].count_weights(); count_w ++)
            File<<list_layers[i][j].get_weight(count_w)<<" ";
            File<<endl;

        }
        File<<endl;
    }

}
net & net::operator = (const net & NET)
{
    n = NET.n;
    Image = NET.Image;
    list_layers.resize(NET.n);
    for (int i = 0; i < NET.n; i++)
        {
            list_layers[i] = NET.list_layers[i];
        }
    exit = NET.exit;
    d_outs = NET.d_outs;
    d_weights = NET.d_weights;
    zero_d_weights = NET.zero_d_weights;
    d_reduce_outs = NET.d_reduce_outs;
    average_weight = NET.average_weight;
    return *this;
}
layer & net::operator[](int n)
{
    return list_layers[n];
}
net::net(const char* filename)
{
    average_weight = 0;
    ifstream File(filename);
    File>>n;
    list_layers.resize(n);
    int size_l;
    vector<vector<double> > v;
    int rec_n, rec_m, sh_n, sh_m, Ker, inp_s, im_n, im_m;
    for (int i = 0; i < n; i++)
    {
        File>>size_l>>rec_n>>rec_m>>sh_n>>sh_m>>Ker>>inp_s>>im_n>>im_m;
        //cout<<size_l<<" "<<rec_n<<" "<<rec_m<<" "<<sh_n<<" "<<sh_m<<" "<<Ker<<" "<<inp_s<<" "<<im_n<<" "<<im_m<<endl;
        list_layers[i].resize(size_l);// создаем первый слой
        v.resize(size_l);
        list_layers[i] = layer(list_layers[i].size(),inp_s, im_n, im_m, rec_n, rec_m, sh_n, sh_m, Ker);
        for (int j = 0; j < size_l; j++)
            v[j].resize(inp_s * rec_n * rec_n + 1);
        for (int k = 0; k < v.size(); k++)
        {
            File>>v[k][0];
            list_layers[i][k].new_zero_weight(v[k][0]);
            for (int j = 1; j < v[k].size(); j++)
            {
                File>>v[k][j];
                list_layers[i][k].new_weight(j-1,v[k][j]);
            }
        }
    }

}
/*void net::dementions_of_exits()
{
    int n,m;
    for (int i = 0; i < list_layers.size(); i++)
    {

    }
}*/
/*void net::remove_layer(int number);
void net::remove_neuron(int position_in_layer, int layers_number);*/
void net::shake(double n)
{
    double sh;
    if (n == -1) sh = average_weight;
    else sh = n;
    for(int i = 0; i < d_weights.size(); i++)
    {

        for (int j = 0; j < d_weights[i].size(); j++)
            {
                double random = (double) (rand() % (20) - 10) *sh;

                list_layers[i][j].change_zero_weights(random);
                for(int q = 0; q < list_layers[i][j].count_weights(); q++)
                {
                    random = (double) (rand() % (20) - 10) * sh;
                    list_layers[i][j].change_weights(q,random);
                }
            }
    }
    //cout<<"average_weight = "<<average_weight*100<<endl;
}
void net::genetic_algo(int iter, int CountOfNet, double per_mutation, vector <maps> & Train, vector<vector<double> > &Targets)
{
    cout<<"Start gen algo"<<endl;
    net best_net;
    vector<net> set_net(CountOfNet);
    vector<double> nets_quality(CountOfNet,0);
    vector<vector<int> > info(this->list_layers.size());
    for (int i = 0; i < info.size(); i++)
    {
        info[i].resize(6);
        info[i][0] = this->list_layers[i].size();
        info[i][1] = this->list_layers[i][0].get_rec_n();
        info[i][2] = this->list_layers[i][0].get_rec_m();
        info[i][3] = this->list_layers[i][0].get_shift_n();
        info[i][4] = this->list_layers[i][0].get_shift_m();
        info[i][5] = this->list_layers[i][0].kernel();
    }
    //create net
    for (int i = 0; i < CountOfNet; i++)
    {
        net TempNet(this->Image,info);
        set_net[i] = TempNet;
    }
    //qality
    for (int i = 0; i < CountOfNet; i++)
    {
        for (int j = 0; j < Train.size(); j++)
        {
            if (set_net[i].result(Train[j])==pos_max(Targets[j])) nets_quality[i]+=1;
        }
        cout<<"quality = "<<nets_quality[i]<<endl;
    }
    best_net = set_net[pos_max(nets_quality)];
    cout<<pos_max(nets_quality)<<endl;
    vector<maps> im(this->list_layers[list_layers.size() - 1].size() * 5);
    vector<vector<double> > tar(this->list_layers[list_layers.size() - 1].size() * 5);
    cout<<"tar size = "<<tar.size()<<endl;
    vector< vector<net> > population(CountOfNet-1);
    vector< vector<double> > pop_rate(CountOfNet-1);
    for (int i = 0 ; i < population.size(); i++ )
    {
        population[i].resize(population.size() - i);
        pop_rate[i].resize(population.size() - i);
    }

    for (int countIter = 0; countIter < iter; countIter++)
    {
        //selection
        for (int bpsize = 0; bpsize < this->exit.size() * 5; bpsize++)
        {
            int random = (int) (rand() % (Targets.size()-1));
            im[bpsize] = Train[bpsize];
            tar[bpsize] = Targets[bpsize];
        }
        for (int i = 0; i < CountOfNet; i++)
            {
                //cout<<"Net "<<i<<" ****************"<<endl;
                set_net[i].back_propogation_offline(tar, im, 0.01, true);
                //set_net[i].back_prop_info();
            }
        //cross
        for (int i = 0 ; i < population.size(); i++ )
        {
            for (int j = 0; j < population[i].size(); j++)
            {
                population[i][j].cross(set_net[i], set_net[i+j+1]);
            }
        }
        for (int i = 0 ; i < population.size(); i++ )
        {
            for (int j = 0; j < population[i].size(); j++)
            {
                //cout<<"pop rate "<<i<<" "<<j<<"\t"<<pop_rate[i][j]<<endl;
                pop_rate[i][j] = 0;
                //cout<<"pop rate "<<i<<" "<<j<<"\t"<<pop_rate[i][j]<<endl;
                for (int k = 0; k < Train.size(); k++)
                {
                    if (population[i][j].result(Train[k])==pos_max(Targets[k])) pop_rate[i][j]+=1;
                }
                cout<<"pop rate "<<i<<" "<<j<<"\t"<<pop_rate[i][j]<<endl;
            }
        }
        //update result
        vector<vector <int> > numBest(2);
        numBest[0].resize(set_net.size());
        numBest[1].resize(set_net.size());
        vector<double> pop_quality(set_net.size());
        for (int k = 0; k < set_net.size(); k++)
        {
            int t_i = 0, t_j = 0;
            double best = 0;
             for (int i = 0 ; i < population.size(); i++ )
            {
                for (int j = 0; j < population[i].size(); j++)
                {
                    //cout<<"pop rate "<<i<<" "<<j<<"\t"<<pop_rate[i][j]<<endl;
                    if (pop_rate[i][j] > best) {best = pop_rate[i][j]; t_i = i; t_j = j;}
                }
            }
            if (k != 0) {
                    if (best == pop_quality[k-1]) {pop_rate[t_i][t_j] = 0; k--;}
                    else {
                        pop_rate[t_i][t_j]  = 0;
                        pop_quality[k] = best;
                        numBest[0][k] = t_i;
                        numBest[1][k] = t_j;
                        }
            }
            else {
                pop_rate[t_i][t_j]  = 0;
                pop_quality[k] = best;
                numBest[0][k] = t_i;
                numBest[1][k] = t_j;
            }
        }
        int ptr = 0;
        for (int i = 0; i < set_net.size(); i++)
        {
            int ch = pos_min(nets_quality);
            if (nets_quality[ch] < pop_quality[ptr])
            {
                //int ch = pos_min(nets_quality);
                set_net[ch] = population[numBest[0][ptr]][numBest[1][ptr]];
                nets_quality[ch] = pop_quality[ptr];
                ptr++;
            }
            /*set_net[i] = population[numBest[0][ptr]][numBest[1][ptr]];
            nets_quality[i] = pop_quality[ptr];
            ptr++;*/
        }
        int worth_n = 0;
        double worth_q = nets_quality[0];
        for (int i = 1; i < set_net.size(); i++)
            if (worth_q > nets_quality[i])  {worth_q = nets_quality[i];  worth_n = i;}
        cout<<"Worth net = "<< worth_n<<endl;
        set_net[worth_n] = net (this->Image,info);
        nets_quality[worth_n] = 0;
        for (int j = 0; j < Train.size(); j++)
            {
                if (set_net[worth_n].result(Train[j])==pos_max(Targets[j])) nets_quality[worth_n]+=1;
            }
        cout<<"Iter "<<countIter<<" :"<<endl;
        for (int i = 0; i < set_net.size(); i++)
            cout<<nets_quality[i]<<endl;
        for (int i = 0; i < set_net.size(); i++)
        {
            double random = (double) (rand() % (1000))/1000;
            if (random < per_mutation)
            {
                cout<<"Mut "<<i<<endl;
                net temp = set_net[i];
                double temp_q =  nets_quality[i];
                set_net[i].shake(0.05);
                nets_quality[i] = 0;
                for (int j = 0; j < Train.size(); j++)
                {
                    if (set_net[i].result(Train[j])==pos_max(Targets[j])) nets_quality[i]+=1;
                }
                if (temp_q > nets_quality[i]) {nets_quality[i] = temp_q; set_net[i] = temp;}
            }
        }

        //mutation
    }

    /*for (int j = 0; j < set_net[0].list_layers[0].size(); j++)
    {
        set_net[0].list_layers[0][j].new_zero_weight(set_net[1].list_layers[0][j].get_zero_weight());
        for(int q = 0; q < set_net[0].list_layers[0][j].count_weights(); q++)
        {
            set_net[0].list_layers[0][j].new_weight(q, set_net[1].list_layers[0][j].get_weight(q));
        }
    }*/


    /*for (int j = 0; j < set_net[0].list_layers[0].size(); j++)
    {
        set_net[0].list_layers[0][j].new_zero_weight(set_net[2].list_layers[0][j].get_zero_weight());
        for(int q = 0; q < set_net[0].list_layers[0][j].count_weights(); q++)
        {
            set_net[0].list_layers[0][j].new_weight(q, set_net[2].list_layers[0][j].get_weight(q));
        }
    }
    nets_quality[0] = 0;

    for (int j = 0; j < Train.size(); j++)
        {
            if (set_net[0].result(Train[j])==pos_max(Targets[j])) nets_quality[0]+=1;
        }*/
}
void net::cross (net & NET1, net & NET2)
{
    n = NET1.n;
    //NET1.back_prop_info();
    Image = NET1.Image;
    list_layers.resize(NET1.n);
    for (int i = 0; i < NET1.n; i++)
        {
            list_layers[i] = NET1.list_layers[i];
        }
    exit = NET1.exit;
    d_outs = NET1.d_outs;
    d_weights = NET1.d_weights;
    zero_d_weights = NET1.zero_d_weights;
    d_reduce_outs = NET1.d_reduce_outs;
    average_weight = NET1.average_weight;
    for (int nl = 0; nl < n; nl++)
        for (int nneur = 0; nneur < list_layers[nl].size(); nneur++)
        {
            double cost_err1 = 0,  cost_err2 = 0;
            //cout<<"d_weights[nl][nneur].size() = "<<d_weights[nl][nneur].size()<<endl;
            for (int nw = 0; nw < d_weights[nl][nneur].size(); nw++)
            {
                //cout<<"NET1.d_weights[nl][nneur][nw] = "<<NET1.d_weights[nl][nneur][nw]<<endl;
                cost_err1 += NET1.d_weights[nl][nneur][nw];
                cost_err2 += NET2.d_weights[nl][nneur][nw];
            }
            cost_err1 += NET1.zero_d_weights[nl][nneur];
            cost_err2 += NET2.zero_d_weights[nl][nneur];
            //double random = (int) (rand() % (Targets.size() -201) +200);
            if (cost_err1 < cost_err2) {
                list_layers[nl][nneur] = NET1.list_layers[nl][nneur];
            }
            else {
                list_layers[nl][nneur] = NET2.list_layers[nl][nneur];
            }
        }
}
net::net()
{
    n = 0;
}
void net::zeroize_diff()
{
    for (int z = 0; z < d_outs.size(); z++)//l
    {
        for (int y = 0; y < d_outs[z].size(); y++)//n
        {
            for (int x = 0; x < d_outs[z][y].n; x++)
            {
                for (int i = 0; i < d_outs[z][y].m; i++)
                d_outs[z][y][x][i] = 0;
            }
        }
    }
    for (int z = 0; z < d_reduce_outs.size(); z++)
    {
        for (int y = 0; y < d_reduce_outs[z].size(); y++)
        {
            for (int x = 0; x < d_reduce_outs[z][y].n; x++)
            {
                for (int i = 0; i < d_reduce_outs[z][y].m; i++)
                d_reduce_outs[z][y][x][i] = 0;
            }
        }
    }

    for (int i = 0; i < d_weights.size(); i++) // l
    {
        for(int j = 0; j < d_weights[i].size(); j++) // n
        {
            for(int k = 0; k < d_weights[i][j].size(); k++)
                d_weights[i][j][k] = 0;
        }
    }
    for (int i = 0; i < zero_d_weights.size(); i++) // l
    {
        for(int j = 0; j < zero_d_weights[i].size(); j++) // n
        {
            zero_d_weights[i][j] = 0;
        }
    }
}
template <typename T>
int pos_max (vector<T> & v)
{
    T max = v[0];
    int pos = 0;
    for (int i = 0; i < v.size(); i++)
        if (max < v[i]) { max = v[i]; pos = i;}
    return pos;
}
template<typename T>
int pos_min (vector<T> & v)
{
    T min = v[0];
    int pos = 0;
    for (int i = 0; i < v.size(); i++)
        if (min > v[i]) { min = v[i]; pos = i;}
    return pos;
}
