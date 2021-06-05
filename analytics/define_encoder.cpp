#include <iostream>
#include <map>
#include <vector>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

auto forward_pass(MatrixXf X, MatrixXf W) {
    auto Y = X * W;
    return Y;
}

auto forward_prop(vector<map<string, MatrixXf>> Layers, vector<string> activations, MatrixXf labels){
    for (auto layer : Layers) {
        auto w = layer["w"];
        cout<<w;
    }
    return 1;
}

int main(){
    MatrixXf x(1, 3);
    x << 1, 3, 4;

    MatrixXf w(3, 2);
    w << 2.3, 3.4, 5.4,
         4.5, 6.7, 2.3;
         

    vector<map<string, MatrixXf>> Layers;
    map<string, MatrixXf> m;
    m["w"] = w;
    Layers.push_back(m);

    vector<string> activations;
    activations.push_back("relu");
    
    MatrixXf labels(1, 1); 
    labels << 1;

    cout<<forward_prop(Layers, activations, labels);
}