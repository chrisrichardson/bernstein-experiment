// @brief Evaluate dofs at quadrature points
// @param dofs0 input dofs
// @param qpts0 Quadrature points (x) for Duffy transform/Stroud quadrature
// @param qpts1 Quadrature points (y)
// @param qvals Values at quadrature points
// @tparam T scalar type
// @tparam N number of dofs in each direction
// @tparam Q number of quadrature points in each direction
// @note input dofs are in the upper left triangle of a 2D array
template <typename T, int N, int Q>
__global__ void evaluate_triangle(const T *dofs0, const T *qpts0, const T *qpts1, T *qvals)
{
    __shared__ T _qpts0[Q], _qpts1[Q];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    if (ty == 0)
    {
        _qpts0[tx] = qpts0[tx];
        _qpts1[tx] = qpts1[tx];
        printf("qpts0[%d]=%f\n", tx, _qpts0[tx]);
        printf("qpts1[%d]=%f\n", tx, _qpts1[tx]);
    }
    __syncthreads();

    __shared__ T c1[N][Q];

    if (tx < N)
        c1[tx][ty] = 0.;

    T p = _qpts0[tx];
    T s = 1.0 - p;
    T r = p / s;
    T w = 1.;
    if (ty < N)
    {
        // T w = s^(N + 1 - ty);
        for (int i = 0; i < N - 1 - ty; ++i)
            w *= s;

        // printf("w[%d, %d]=%f\n", tx, ty, w);

        for (int alpha2 = 0; alpha2 < N - ty; ++alpha2)
        {
            // c1[ty][tx] += w;
            c1[ty][tx] += w * dofs0[ty + alpha2 * N];
            w *= r * (N - 1 - ty - alpha2) / (1 + alpha2);
        }
        // printf("c1[%d, %d]=%f\n", tx, ty, c1[ty][tx]);
    }

    T qval = 0.;

    __syncthreads();

    p = _qpts1[tx];
    s = 1.0 - p;
    r = p / s;

    w = 1.;
    for (int i = 0; i < N - 1; ++i)
        w *= s;
    for(int alpha1 = 0; alpha1 < N; ++alpha1) {
        qval += w * c1[alpha1][ty];
        w *= r * (N - 1 - alpha1) / (1 + alpha1);
    }

    qvals[tx + ty*Q] = qval;
}


// @brief Provide dofs values from values at quadrature points
// Pre: tx [0...Q]
// // Pre: ty [0...Q]
// template <typename T, int N, int Q>
// __global__ void triangle_moment(const T *qpts0, const T *qpts1, const T *qwts0, const T *qwts1, T *dofs0)
// {
//     __shared__ T _qpts0[Q], _qpts1[Q], _qwts0[Q], _qwts1[Q];
//     const int tx = threadIdx.x;
//     const int ty = threadIdx.y;

//     if (ty == 0)
//     {
//         _qpts0[tx] = qpts0[tx];
//         _qpts1[tx] = qpts1[tx];
//         _qwts0[tx] = qwts0[tx];
//         _qwts1[tx] = qwts1[tx];
//         printf("qpts0[%d]=%f\n", tx, _qpts0[tx]);
//         printf("qpts1[%d]=%f\n", tx, _qpts1[tx]);
//         printf("qwts0[%d]=%f\n", tx, _qwts0[tx]);
//         printf("qwts1[%d]=%f\n", tx, _qwts1[tx]);

//     }
//     __syncthreads();

//     __shared__ T c1[N][Q];

//     if (tx < N)
//         c1[tx][ty] = 0.;

//     T p = _qpts0[tx];
//     T s = 1.0 - p;
//     T r = p / s;
//     T w = 1.;
//     if (ty < N)
//     {
//         // T w = s^(N + 1 - ty);
//         for (int i = 0; i < N - 1 - ty; ++i)
//             w *= s;

//         // printf("w[%d, %d]=%f\n", tx, ty, w);

//         for (int alpha2 = 0; alpha2 < N - ty; ++alpha2)
//         {
//             // c1[ty][tx] += w;
//             c1[ty][tx] += w * dofs0[ty + alpha2 * N];
//             w *= r * (N - 1 - ty - alpha2) / (1 + alpha2);
//         }
//         // printf("c1[%d, %d]=%f\n", tx, ty, c1[ty][tx]);
//     }

//     T qval = 0.;

//     __syncthreads();

//     p = _qpts1[tx];
//     s = 1.0 - p;
//     r = p / s;

//     w = 1.;
//     for (int i = 0; i < N - 1; ++i)
//         w *= s;
//     for(int alpha1 = 0; alpha1 < N; ++alpha1) {
//         qval += w * c1[alpha1][ty];
//         w *= r * (N - 1 - alpha1) / (1 + alpha1);
//     }

//     qvals[tx + ty*Q] = qval;
// }