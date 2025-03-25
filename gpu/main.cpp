#include <iostream>
#include <fstream>
#include <format>
#include <string>

#include <thrust/device_vector.h>

#include "bernstein.h"

template <typename T>
void load_quadrule(T& qpts0, T& qpts1, T& qwts0, T& qwts1, int Q)
{
    std::ifstream quad_rule(std::format("{}/quadrules/{}.txt", PROJECT_SOURCE_DIR, Q));
    std::string line;
    for (int q = 0; q < Q; q++)
    { // X Quad Nodes
        getline(quad_rule, line, ' ');
        qpts0[q] = stod(line);
    }
    getline(quad_rule, line); 
    for (int q = 0; q < Q; q++)
    { // X Quad Weights
        getline(quad_rule, line, ' ');
        qwts0[q] = stod(line);
    }
    getline(quad_rule, line); 
    
    for (int q = 0; q < Q; q++)
    { // Y Quad Nodes
        getline(quad_rule, line, ' ');
        qpts1[q] = stod(line);
    }
    getline(quad_rule, line); 

    
    for (int q = 0; q < Q; q++)
    { // Y Quad Weights
        getline(quad_rule, line, ' ');
        qwts1[q] = stod(line);
    }
    // getline(quad_rule, line); 
}

int main()
{
    const int Q = 4;
    const int order = 2;
    const int N = order + 1;

    std::vector<double> qpts0(Q), qwts0(Q);
    std::vector<double> qpts1(Q), qwts1(Q);

    
    std::vector<double> dofs = {
        1.0, 1.0, 1.0,
        1.0, 1.0, 0.0,
        1.0, 0.0, 0.0,
    };

    // std::vector<double> dofs = {
    //     1.0, 0.5, 0.5,
    //     0.5, 0.5, 0.0,
    //     0.5, 0.0, 0.0,
    // };

    std::vector<double> qvals(Q * Q);
    
    // read quadrature points from quad_rules
    load_quadrule(qpts0, qpts1, qwts0, qwts1, Q);
    
    // for (int i = 0; i < Q; i++)
    // {
    //     std::cout << qwts0[i] << " ";
    // }
        
    // for (int i = 0; i < Q; i++)
    // {
    //     std::cout << qwts1[i] << " ";
    // }
    

    thrust::device_vector<double> qpts0_d(qpts0.begin(), qpts0.end());
    thrust::device_vector<double> qpts1_d(qpts1.begin(), qpts1.end());
    thrust::device_vector<double> qwts0_d(qwts0.begin(), qwts0.end());
    thrust::device_vector<double> qwts1_d(qwts1.begin(), qwts1.end());
    thrust::device_vector<double> dofs_d(dofs.begin(), dofs.end());
    thrust::device_vector<double> qvals_d(qvals.size());

    dim3 grid_size(1);
    dim3 block_size(Q, Q);
    evaluate_triangle<double, N, Q><<<grid_size, block_size>>>(thrust::raw_pointer_cast(dofs_d.data()),
                                                 thrust::raw_pointer_cast(qpts0_d.data()),
                                                 thrust::raw_pointer_cast(qpts1_d.data()),
                                                 thrust::raw_pointer_cast(qvals_d.data()));

    thrust::copy(qvals_d.begin(), qvals_d.end(), qvals.begin());
    for (int i = 0; i < Q; i++)
    {
        for (int j = 0; j < Q; j++)
        {
            std::cout << qvals[i + j * Q] << " ";
        }
        std::cout << "\n";
    }

    triangle_moment<double, N, Q><<<grid_size, block_size>>>(
        thrust::raw_pointer_cast(qpts0_d.data()),
        thrust::raw_pointer_cast(qpts1_d.data()),
        thrust::raw_pointer_cast(qwts0_d.data()),
        thrust::raw_pointer_cast(qwts1_d.data()),
        thrust::raw_pointer_cast(qvals_d.data()),
        thrust::raw_pointer_cast(dofs_d.data())
    );

    return 0;
}