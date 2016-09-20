#include "Headers.h"
#include <cmath>
using namespace std;
using namespace activation_functions;

net::net(maps IMAGE, vector<vector<int> > Info)
{
    n = Info.size();// количество слоев
    list_layers.resize(n);
    list_layers[0].resize(Info[0][0]);// создаем первый слой
    vector<maps*> Im;
    Image = IMAGE;
    Im.push_back(&Image);


    cout<<"\n zero layer"<<endl;
    list_layers[0] = layer(list_layers[0].size(), th, Info[0][1],Info[0][2], Im, Info[0][3], Info[0][4], Info[0][5]);
    list_layers[0].info();
    vector<maps*> temp_feachure_maps;

    for (int i = 1; i < n; i++) // создаем остальные слои
    {
       // temp_feachure_maps.resize(list_layers[i-1].size());
        temp_feachure_maps = list_layers[i-1].Get_Outs();
        layer Temp(Info[i][0], th, Info[i][1], Info[i][2], temp_feachure_maps, Info[i][3], Info[i][4], Info[i][5]);
        Temp.info();
        list_layers[i] = Temp;
    }
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
    //info();

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
    //cout<<"\nflag0"<<endl;
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
            int sh_n = list_layers[nl+1][0].get_shift_n(), sh_m = list_layers[nl+1][0].get_shift_m();///change
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
                    d_outs[nl][i][k][m] = 0;
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

           // int n_next = d_outs[nl][0].n, m_next = d_outs[nl][0].m;//sizes of this maps(changed)

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

    // change weights
    for(int i = 0; i < d_weights.size(); i++)
    {

        for (int j = 0; j < d_weights[i].size(); j++)
            {
                //speed +=0.01;
                int n_next, m_next;
                if (list_layers[i].is_reduce())
                {
                    n_next = d_reduce_outs[i][0].n; m_next = d_reduce_outs[i][0].m;//sizes of next maps
                }
            else
                {
                    n_next = d_outs[i][0].n; m_next = d_outs[i][0].m;//sizes of next maps
                }
                list_layers[i][j].change_zero_weights(0);
                for(int q = 0; q < list_layers[i][j].count_weights(); q++)
                list_layers[i][j].change_weights(q,speed*d_weights[i][j][q]);
            }
    }


    //info();
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
        }

        for (int j = 0; j < list_layers[i].size(); j++)
        {
            list_layers[i][j].change_inputs(temp_feachure_maps);
            list_layers[i][j].convolution();
            list_layers[i][j].get_exit();
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
            //cout<<"ok"<<endl;
        }

        for (int j = 0; j < list_layers[i].size(); j++)
        {
            list_layers[i][j].change_inputs(temp_feachure_maps);
            //list_layers[i][j].info_inputs();
            list_layers[i][j].activate();
            list_layers[i][j].convolution();
            //cout<<"ok2"<<endl;
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
            for (int j = 0; j < list_layers[i].size(); j++)
                {
                    list_layers[i][j].info_inputs();
                    cout<<endl;
                    list_layers[i][j].info_feature();
                    cout<<endl;
                }

            }

   /* cout<<"\nWeights"<<endl;
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
void net::add_layer(int number, int count_neurons)
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
net & net::operator = (net const& NET)
{
    /*for (int i = 0; i < NET.net_size(); i++)
        list_layers = NET[i];
    return *this;*/
}
layer & net::operator[](int n)
{
    return list_layers[n];
}
net::net(const char* filename)
{
    ifstream File(filename);
    File>>n;
    //cout<<n<<endl;
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
            //cout<<v[k][0]<<endl;
            list_layers[i][k].new_zero_weight(v[k][0]);
            for (int j = 1; j < v[k].size(); j++)
                {
                    File>>v[k][j];
                    //cout<<v[k][j]<<" ";
                    list_layers[i][k].new_weight(j-1,v[k][j]);
                    //cout<<v[k][j]<<endl;
                }
            //cout<<endl;
        }
        //cout<<"Layer"<<i<<" -ok"<<endl;
    }

    //list_layers[0].info();
    //vector<maps*> temp_feachure_maps;

    /*for (int i = 1; i < n; i++) // создаем остальные слои
    {
       // temp_feachure_maps.resize(list_layers[i-1].size());
        temp_feachure_maps = list_layers[i-1].Get_Outs();
        layer Temp(Info[i][0], th, Info[i][1], Info[i][2], temp_feachure_maps, Info[i][3], Info[i][4], Info[i][5]);
        Temp.info();
        list_layers[i] = Temp;
    }*/

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
