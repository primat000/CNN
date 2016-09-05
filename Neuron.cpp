#include "Headers.h"
using namespace std;

/*................class neuron...............*/
neuron::neuron(double (*F_T)(double),int r_n, int r_m, vector<maps*> M, int shift_n, int shift_m, int Kernel_n, int Kernel_m)
{

    rec_n = r_n;
    rec_m = r_m;
    weights.resize(rec_n*rec_m*M.size());
    //cout<<"rec n = "<<r_n<<" rec m = "<<r_m<<" size ="<<M.size()<<endl;
    weight_ini();
    function_type = F_T;
    inputs = M;
    step =1;/// delete
    kernel_n = Kernel_n;  //////////////default = 1
    kernel_m = Kernel_m;  //////////////default = 1
    EXIT = 0;
    sh_n = shift_n; //////////////default = 1
    sh_m = shift_m; //////////////default = 1

}
void neuron::weight_ini()
{
    srand(time(0));
    for (int i = 0; i < weights.size(); i++)
        {
            weights[i] = (rand() % 1000)/1000.0;
            //cout<<"w ="<<weights[i]<<endl;
        }
   //cout<<endl;
}

double neuron::function(double x)
{
    return function_type(x);
}

maps* neuron::get_exit()
{
    /*for (int i = 0; i < feature_map.size(); i++)
    {
        for (int l = 0; l < feature_map[0].size(); l++)
            cout<<feature_map[i][l]<<"  ";
        cout<<endl;
    }*/
    if (feature_map.size() == feature_map[0].size() == 1) {EXIT = feature_map[0][0];}
    maps temp(feature_map);
    exit = temp;
    return &exit;

}

void neuron::activate()
{
    int v = inputs.size();
    int n = inputs[0]->n;
    int m = inputs[0]->m;
    vector<double> field;
    int new_n = 0, new_m = 0;
    int feature_map_size_n = (n - rec_n)/sh_n + 1, feature_map_size_m = (m - rec_m)/sh_m + 1;
    //cout<<"\nfeature_map_size_n = "<< feature_map_size_n<< "\t feature_map_size_m ="<<feature_map_size_m;
    /*feature_map.resize(n - rec_n +1);
    Sigma.resize(n - rec_n + 1);*/
    feature_map.resize(feature_map_size_n);
    Sigma.resize(feature_map_size_n);
    /*for (int i = 0; i < feature_map.size(); i++)
    {
        feature_map[i].resize(m - rec_m +1);
        Sigma[i].resize(m-rec_m + 1);
    }*/
    for (int i = 0; i < feature_map_size_n; i++)
    {
        feature_map[i].resize(feature_map_size_m);
        Sigma[i].resize(feature_map_size_m);
    }

    for (int it = 0; it < (feature_map_size_n)* (feature_map_size_m); it++)
    {
        for (int k = 0; k < v; k++)
        {
            maps temp_map = *inputs[k];
            for (int i = new_n; i < new_n + rec_n; i++ )
            {
                for (int j = new_m; j < new_m + rec_m; j++)
                {
                    field.push_back(temp_map[i][j]);
                    //cout<<"t_m = "<<temp_map[i][j]<<" ";
                }
                //cout<<endl;
            }
        }
        feature_map[new_n][new_m]=0;
        for (int l = 0; l < field.size(); l++)
                {
                    feature_map[new_n][new_m]+=weights[l]*field[l];
                }
        Sigma[new_n][new_m] = feature_map[new_n][new_m];
        feature_map[new_n][new_m] = this->function(Sigma[new_n][new_m]);

        new_m+=step;
        if (m - new_m < rec_m) new_n ++;
        if (m - new_m < rec_m) new_m = 0;

                field.clear();
    }

}

void neuron::convolution ()
{
    int v = inputs.size(); // количество карт подаваемых на вход
    int n = inputs[0]->n; // высота
    int m = inputs[0]->m; // ширина
    vector<double> field;
    int new_n = 0, new_m = 0;
    for (int it = 0; it < (n - rec_n +1)* (m - rec_m +1); it++)
    {
        for (int k = 0; k < v; k++)
        {
            maps temp_map = *inputs[k];
            for (int i = new_n; i < new_n + rec_n; i++ )
            {
                for (int j = new_m; j < new_m + rec_m; j++)
                {
                    field.push_back(temp_map[i][j]);
                }
            }
        }
        feature_map[new_n][new_m]=0;
        for (int l = 0; l < field.size(); l++)
                {
                    feature_map[new_n][new_m]+=weights[l]*field[l];
                }
        Sigma[new_n][new_m] = feature_map[new_n][new_m];
        feature_map[new_n][new_m] = this->function(Sigma[new_n][new_m]);

        new_m+=step;
        if (m - new_m < rec_m) new_n ++;
        if (m - new_m < rec_m) new_m = 0;

                field.clear();
    }
    /*for (int i = 0; i < feature_map.size(); i++)
    {
        for (int l = 0; l < feature_map[0].size(); l++)
            cout<<feature_map[i][l]<<"  ";
        cout<<endl;
    }*/

}

void neuron::change_inputs(vector<maps*> new_inputs)
{
    if (inputs.size() != new_inputs.size()) {cout<<"Dimentions incorrect!\n"<<endl; return;}
    for(int i = 0; i < new_inputs.size(); i++)
        inputs[i] = new_inputs[i];
}

double neuron::get_EXIT ()
{
    return EXIT;
}

int neuron::count_weights()
{
    return weights.size();
}
int neuron::feature_map_n()
{
    return feature_map.size();
}

int neuron::feature_map_m()
{
    return feature_map[0].size();
}
int neuron::inputs_n(int i)
{
    return inputs[i]->n;
}
int neuron::inputs_m(int i)
{
    return inputs[i]->m;
}
vector<maps*> & neuron::get_inputs()
{
    //cout<<"ok"<<endl;
    return inputs;
}
void neuron::change_weights(int n, double dw)
{
    weights[n]-=dw;
}
int neuron::get_rec_n()
{
    return rec_n;
}
int neuron::get_rec_m()
{
    return rec_m;
}
double neuron::get_weight(int n)
{
    return weights[n];
}
void neuron::info_inputs()
{
    for (int i = 0; i < inputs.size(); i++)
    {
        maps temp = *inputs[i];
        cout<<"inputs "<<i<<":"<<endl;
        for (int k = 0; k < temp.n; k++)
        {
            for (int m = 0; m <temp.m; m++)
            cout<<temp[k][m]<<" ";
            cout<<endl;
            ;
        }
    }

}

int neuron::count_of_maps()
{
    return inputs.size();
}

/*................class map...............*/
vector<double> & maps::operator[](int n)
{
    return map[n];
}
maps::maps(const double* mas,int n,int m)
{
    map.resize(n);
    for (int i = 0; i < n; i++)
        map[i].resize(m);

    this->n = n;
    this->m = m;
    for (int i = 0; i < n; i++)
    {
        for(int j = 0; j < m; j++)
        {
            this->map[i][j] = mas[i*m + j];
            //cout<<mas[i*m + j]<<" ";
        }
        //cout<<endl;
    }

}
maps::maps(vector<vector<double> > mm)
{
    map = mm;
    n = mm.size();
    m = mm[0].size();
}
maps::maps()
{
    n=0;
    m=0;
}
const maps& maps::operator = (const maps &M)
{
    n = M.n;
    m = M.m;
    map.resize(n);
    for (int i = 0; i < n; i++)
        map[i].resize(m);
    for(int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
        map[i][j] = M.map[i][j];
    return *this;
}

void maps::map_resize(int N, int M)
{
    n = N;
    m = M;
    map.resize(n);
    for (int i = 0; i < n; i++)
        map[i].resize(m);
}
