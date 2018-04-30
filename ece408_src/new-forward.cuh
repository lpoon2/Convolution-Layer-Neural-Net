
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
/*
  This will be refractor after Wally's optimization (writing weight matrix into constant memory)
*/
__global__ void unroll_weight(float *output_w, const float *k, const int M, const int C, const int K) {
    int m = (blockIdx.y * blockDim.y + threadIdx.y) % M;
    int c = (blockIdx.x * blockDim.x + threadIdx.x) / (K*K);
    int h = (blockIdx.x * blockDim.x + threadIdx.x) / K;
    int w = (blockIdx.x * blockDim.x + threadIdx.x) % K;
    #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    if ((blockIdx.x * blockDim.x + threadIdx.x) < (K*K*C)) && ((blockIdx.y * blockDim.y + threadIdx.y) < M) {
      int idx = (K*K*C) * (blockIdx.y * blockDim.y + threadIdx.y) + (blockIdx.x * blockDim.x + threadIdx.x);
      output_w[idx] = k4d(m,c,h,w);
    }
    #undef k4d
}
__global__ void unroll_input(float *output_x, const float *x) {
  #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
  int start_h = ((blockIdx.x * blockDim.x + threadIdx.x) % (H*W)) / K;
  int start_w = (((blockIdx.x * blockDim.x + threadIdx.x) % (H*W)) % K;
  int h = start_h + ((blockIdx.y * blockDim.y + threadIdx.y) % (K*K)) / K;
  int w = start_w + ((blockIdx.y * blockDim.y + threadIdx.y) % (K*K)) % K;
  int c = (blockIdx.y * blockDim.y + threadIdx.y) / (K*K);
  int b = (blockIdx.x * blockDim.x + threadIdx.x) / (H*W);

  if ((blockIdx.x * blockDim.x + threadIdx.x) < (H*W*B)) && ((blockIdx.y * blockDim.y + threadIdx.y) < (H*W*C)) {
    int idx = (H*W*B) /* make sure our gridDim and block dim match with this*/
    * (blockIdx.y * blockDim.y + threadIdx.y) + (blockIdx.x * blockDim.x + threadIdx.x);
    output_x[idx] = x4d(b,c,h,w);
  }

  #undef x4d
}

__global__ void unroll_multipy(float *weight, float *input, float output) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int weight_wid = K*K*C;
  int input_wid = H*W*B;
  int b = col / (H*W);
  int m = row;
  int h = (col % (H*W)) / W;
  int w = (col % (H*W)) % W;
  float temp = 0;
  for (int i = 0; i< weight_wid; i++) {
      temp += weight[row * weight_wid + i] * input[i * input_wid + col];
  }
  y4d(b,m,h,w) = temp;
}

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
