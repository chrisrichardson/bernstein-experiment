#include <iostream>
#include <fstream>
#include <format>
#include <string>

#include <thrust/device_vector.h>

#include "bernstein.h"

template <typename T>
void load_quadrule(T &qpts0, T &qpts1, T &qwts0, T &qwts1, int Q)
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

template <typename T, int N, int Q>
std::vector<double> compute_mass_matrix_triangle(auto f, const T &qpts0, const T &qpts1, const T &qwts0, const T &qwts1)
{
    // Get qvals: evaluate f at quad points
    std::vector<double> qvals(Q*Q);
    for(int q0 = 0; q0 < Q; ++q0) {
        for(int q1 = 0; q1 < Q; ++q1) {
            qvals[q0 + Q*q1] = f(qpts0[q0], qpts1[q1]);   
        }
    }
    thrust::device_vector<double> qvals_d(qvals.begin(), qvals.end());
    thrust::device_vector<double> dofs_d(dofs.size());

    // moments = compute_moments_triangle(2 * n, f, fdegree)
    triangle_moment<double, 2*N, Q><<<grid_size, block_size>>>(
        thrust::raw_pointer_cast(qpts0_d.data()),
        thrust::raw_pointer_cast(qpts1_d.data()),
        thrust::raw_pointer_cast(qwts0_d.data()),
        thrust::raw_pointer_cast(qwts1_d.data()),
        thrust::raw_pointer_cast(qvals_d.data()),
        thrust::raw_pointer_cast(dofs_d.data()));
    

    
    // mat = np.zeros(((n + 1) * (n + 2) // 2, (n + 1) * (n + 2) // 2))

    // i = 0
    // for a in range(n + 1):
    //     for b in range(n + 1 - a):
    //         j = 0
    //         for c in range(n + 1):
    //             for d in range(n + 1 - c):
    //                 mat[i, j] = multichoose([b + d, a + c], [b, a])
    //                 mat[i, j] /= choose(2 * n, n)
    //                 mat[i, j] *= moments[b + d, a + c]
    //                 j += 1
    //         i += 1

    // return mat
}

int main()
{
    const int Q = 4;
    const int order = 2;
    const int N = order + 1;

    std::vector<double> qpts0(Q), qwts0(Q);
    std::vector<double> qpts1(Q), qwts1(Q);


    // read quadrature points from quad_rules
    load_quadrule(qpts0, qpts1, qwts0, qwts1, Q);

    // Simple one element test
    std::vector<double> dofs = {
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
    };
    std::vector<double> qvals(Q * Q);

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
        thrust::raw_pointer_cast(dofs_d.data()));

    thrust::copy(dofs_d.begin(), dofs_d.end(), dofs.begin());
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            std::cout << dofs[i + j * N] << " ";
        }
        std::cout << "\n";
    }

    // Mass Matrix
    auto f = [](double x, double y){return x*x + y*y;};
    compute_mass_matrix_triangle<double, N, Q>(f, qpts0, qpts1, qwts0, qwts1);

    return 0;
}