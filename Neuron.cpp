#include "Headers.h"
using namespace std;

/*................class neuron...............*/
neuron::neuron(double (*F_T)(double),int r_n, int r_m, vector<maps*> M, int shift_n, int shift_m, int K)
{

    rec_n = r_n;
    rec_m = r_m;
    Kernel = K;
    weights.resize(rec_n*rec_m*M.size());
    //cout<<"rec n = "<<r_n<<" rec m = "<<r_m<<" size ="<<M.size()<<endl;
    weight_ini();
    function_type = F_T;
    inputs = M;
    step =1;/// delete
    EXIT = 0;
    sh_n = shift_n; //////////////default = 1
    sh_m = shift_m; //////////////default = 1

}
void neuron::weight_ini()
{

    for (int i = 0; i < weights.size(); i++)
        {
            weights[i] = (rand() % 5000)/10000.0;
            //cout<<"w ="<<weights[i]<<endl;
        }
    w0 = (rand() % 5000)/10000.0;
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
    return &fea;

}
void neuron::info_feature()
{
    cout<<"feature map sizes n = "<<feature_map_n()<<"\t m = "<<feature_map_m()<<endl;
    for (int i = 0; i < feature_map.size(); i++)
    {
        for (int l = 0; l < feature_map[0].size(); l++)
            cout<<feature_map[i][l]<<"  ";
        cout<<endl;
    }
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

    feature_map.resize(feature_map_size_n);
    Sigma.resize(feature_map_size_n);

    for (int i = 0; i < feature_map_size_n; i++)
    {
        feature_map[i].resize(feature_map_size_m);
        Sigma[i].resize(feature_map_size_m);
    }

    for (int in = 0; in < feature_map_size_n; in++)
    {
        for (int im = 0; im < feature_map_size_m; im++)
        {
            Sigma[in][im] = 0;
            for (int k = 0; k < v; k++)
            {
                maps temp_map = *inputs[k];
                for (int i = in * sh_n; i < in * sh_n + rec_n; i++ )
                {
                    for (int j = im * sh_m; j < im * sh_m + rec_m; j++)
                    {
                        Sigma[in][im]+= temp_map[i][j] * weights[k * rec_n * rec_m + rec_m * (i - in * sh_n) + j - im * sh_m];
                    }
                }
            }
            Sigma[in][im] += w0;
            feature_map[in][im] = this->function(Sigma[in][im]);
        }
    }
}

void neuron::convolution ()
{
    int v = inputs.size(); // ���������� ���� ���������� �� ����
    int n = inputs[0]->n; // ������
    int m = inputs[0]->m; // ������
    vector<double> field;
    int new_n = 0, new_m = 0;
     for (int in = 0; in < feature_map_n(); in++)
        {
            for (int im = 0; im < feature_map_m(); im++)
            {
                Sigma[in][im] = 0;
                for (int k = 0; k < v; k++)
                {
                    maps temp_map = *inputs[k];
                    for (int i = in * sh_n; i < in * sh_n + rec_n; i++ )
                    {
                        for (int j = im * sh_m; j < im * sh_m + rec_m; j++)
                        {
                            Sigma[in][im]+= temp_map[i][j] * weights[k * rec_n * rec_m + rec_m * (i - in * sh_n) + j - im * sh_m];
                            //cout<<"\nweights = "<<k * rec_n * rec_m + rec_m * (i - in * sh_n) + j - im * sh_m;
                        }
                    }
                }
                Sigma[in][im] += w0;
                feature_map[in][im] = this->function(Sigma[in][im]);
            }
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
void neuron::change_zero_weights(double dw)
{
    w0 -= dw;
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
double neuron::get_zero_weight()
{
    return w0;
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

int neuron::get_shift_n()
{
    return sh_n;
}

int neuron::get_shift_m()
{
    return sh_m;
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


