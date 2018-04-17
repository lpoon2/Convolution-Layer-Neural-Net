#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_
#define TITLE_SIZE 32
#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

__global__ void buildWeightMatrix(const float *k, float *weight_mat, int K /*weight tile size*/ ,const int M, const int C) {
  int row = blockIdx.y * M + blockIdx.x;
  int b = blockIdx.y;
  int c = blockIdx.x;
  int i = threadIdx.y * TITLE_SIZE + threadIdx;
  if (i < K*K) {
    weight_mat[(b+c)*(K*K)+i] = k4d(b, c, (i / K) % K, i % K);
  }
  #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
}

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int W_grid = W_out / TITLE_SIZE;
    int H_grid = H_out / TITLE_SIZE;
    int m = blockIdx.x;
    int h = blockIdx.y / W_grid + threadIdx.y;
    int w = blockIdx.y % W_grid + threadIdx.x;

    (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    (void)W_out; // silence declared but never referenced warning. remove this line when you start working

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]



#undef y4d
#undef x4d
#undef k4d
}

/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];

    const int H_out = H - K + 1;
    const int W_out = W - K + 1

    const int W_wmat = K*K*C; /* width of weight matrix*/
    const int H_wmat = M; /* height */
    const int W_imat = H_out * W_out;
    const int H_imat = K*K*C;

    int W_grid = W_out / TITLE_SIZE;
    int H_grid = H_out / TITLE_SIZE;
    const int Y = H_grid * W_grid;
    float *k;
    /*
    Allocate GPU memory
    */
    cudaMalloc((void**) &k, sizeof(float) * B * C * K * K);
    /*
    Copy input memory into GPU memory
    */
    cudaMemcpy(k, w, sizeof(float) * B * C * K * K, cudaMemcpyHostToDevice);

    dim3 blockDim(ceil( K / TILE_WIDTH), ceil( K / TILE_WIDTH), 1);
    dim3 gridDim(M, Y, 1);
    buildWeightMatrix<<gridDim, blockDim>>();
    // Set the kernel dimensions
    // dim3 gridDim(0);
    // dim3 blockDim(0);

    // Call the kernel
    // forward_kernel<<<gridDim, blockDim, 0, s>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}

/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif
