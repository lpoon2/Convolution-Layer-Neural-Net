
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_
#define TILE_WIDTH 24
#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

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
    int W_grid=ceil((double)W_out/TILE_WIDTH);
    int H_grid=ceil((double)H_out/TILE_WIDTH);


// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
int weight_size=C*K*K;
int input_size=C*H*W;
extern __shared__ float shmem[];
float* X_shared = &shmem[0];
float* W_shared = &shmem[weight_size]; //C*K*K =X_TILE_WIDTH*X_TILE_WIDTH

int idx=threadIdx.x*TILE_WIDTH+threadIdx.y; //load shared mem
if(weight_size>idx){
  X_shared[idx]=k[blockIdx.y*(weight_size)+idx];
}
do{
  if(input_size>idx){
  W_shared[idx]=x[blockIdx.x*(input_size)+idx];
  }
  idx+=TILE_WIDTH*TILE_WIDTH;
}while(idx<input_size);
__syncthreads();
float sum=0;
for(int c=0;c<C;c++){
  for(int p=0;p<K;p++) {
    for(int q=0; q<K; q++){
      sum+=W_shared[c*(H*W)+(threadIdx.x+p)*W+threadIdx.y+q]*X_shared[c*(K*K)+p*K+q];
    }
  }
}
y4d(blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y)=sum;
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
    //CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";
    cudaStream_t s = y.stream_->stream_;
    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
    const int B = x.shape_[0]; //batches
    const int M = y.shape_[1]; //output channels
    const int C = x.shape_[1]; //input channels
    const int H = x.shape_[2]; //height of input
    const int W = x.shape_[3]; //width of input
    const int K = w.shape_[3]; //height and width of weights
    // Set the kernel dimensions
     dim3 gridDim(B,M,1);
     dim3 blockDim(TILE_WIDTH,TILE_WIDTH,1);
     size_t shmem_size = sizeof(float) * ( (K*K + H*W)*C );
    // Call the kernel
    forward_kernel<<<gridDim, blockDim, shmem_size, s>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

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