
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_
#define TILE_WIDTH 32

#include <mxnet/base.h>
#include <stdio.h>
#include <math.h>

namespace mxnet
{
namespace op
{

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int b = (blockIdx.y * blockDim.y + threadIdx.y) / H_out;
    //int b = blockIdx.y;
    int m = (blockIdx.x * blockDim.x + threadIdx.x) / W_out;
    //int m = blockIdx.x;
    int h = (blockIdx.y * blockDim.y + threadIdx.y) % H_out;
    int w = (blockIdx.x * blockDim.x + threadIdx.x) % W_out;

    if (((blockIdx.x * blockDim.x + threadIdx.x) < (W_out * M)) && ((blockIdx.y * blockDim.y + threadIdx.y) < (H_out * B))) {
      float acc=0;
      for(int c=0;c<C;c++) {
        for(int p=0;p<K;p++) {
          for(int q=0; q<K; q++) {
            acc+=x4d(b,c,h+p,w+q)*k4d(m,c,p,q);
          }
        }
      }
      y4d(b,m,h,w)=acc;
    }
    //y4d(blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y)=sum;
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
    const int B = x.shape_[0]; //batches
    const int M = y.shape_[1]; //output channels
    const int C = x.shape_[1]; //input channels
    const int H = x.shape_[2]; //height of input
    const int W = x.shape_[3]; //width of input
    const int K = w.shape_[3]; //height and width of weights
    // Set the kernel dimensions
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int X = ceil((M*W_out)/(TILE_WIDTH*1.0));
    int Y = ceil((B*H_out)/(TILE_WIDTH*1.0));
    dim3 gridDim(X, Y, 1);
    dim3 blockDim(TILE_WIDTH,TILE_WIDTH,1);
    // Call the kernel
    forward_kernel<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);
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
