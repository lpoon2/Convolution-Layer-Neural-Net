
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
  int m = (blockIdx.y * blockDim.y + threadIdx.y);
  int c = (blockIdx.x * blockDim.x + threadIdx.x) / (K*K);
  int h = ((blockIdx.x * blockDim.x + threadIdx.x) % (K*K)) / K;
  int w = (blockIdx.x * blockDim.x + threadIdx.x) % K;
  //int weight_wid = TILE_WIDTH * ceil((C * K * K)/(TILE_WIDTH*1.0));
  int idx = (C*K*K) * (blockIdx.y * blockDim.y + threadIdx.y) + (blockIdx.x * blockDim.x + threadIdx.x);
  #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

  if (((blockIdx.x * blockDim.x + threadIdx.x) < (K*K*C)) && ((blockIdx.y * blockDim.y + threadIdx.y) < M)) {
    output_w[idx] = k4d(m,c,h,w);
  }
  __syncthreads();
  #undef k4d
}

__global__ void unroll_input(float *output_x, const float *x, const int B, const int C, const int H, const int W, const int K) {
  #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
  const int H_out = H - K + 1;
  const int W_out = W - K + 1;
  int start_h = ((blockIdx.x * blockDim.x + threadIdx.x) % (H_out*W_out)) / W_out;
  int start_w = ((blockIdx.x * blockDim.x + threadIdx.x) % (H_out*W_out)) % W_out;
  int h = start_h + (((blockIdx.y * blockDim.y + threadIdx.y) % (K*K)) / K);
  int w = start_w + (((blockIdx.y * blockDim.y + threadIdx.y) % (K*K)) % K);
  int c = (blockIdx.y * blockDim.y + threadIdx.y) / (K*K);
  int b = (blockIdx.x * blockDim.x + threadIdx.x) / (H_out*W_out);
  //int input_wid = TILE_WIDTH * ceil((H*W*B)/(TILE_WIDTH*1.0));
  int idx = (H_out*W_out*B) * (blockIdx.y * blockDim.y + threadIdx.y) + (blockIdx.x * blockDim.x + threadIdx.x);

  if (((blockIdx.x * blockDim.x + threadIdx.x) < (H_out*W_out*B)) && ((blockIdx.y * blockDim.y + threadIdx.y) < (K*K*C))) {
    output_x[idx] = x4d(b,c,h,w);
  }
  __syncthreads();
  #undef x4d
}

__global__ void unroll_multipy(float *weight, float *input, float *y, const int B, const int M, const int C, const int H, const int W, const int K) {
  const int H_out = H - K + 1;
  const int W_out = W - K + 1;
  #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int weight_wid = K*K*C;
  int input_wid = H_out*W_out*B;
  int b = col / (H_out*W_out);
  int m = row;
  int h = (col % (H_out*W_out)) / W_out;
  int w = (col % (H_out*W_out)) % W_out;
  //float temp = 0;

  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
  float Pvalue = 0.0 ;

  for(int ph = 0 ; ph < (weight_wid-1)/TILE_WIDTH + 1 ; ph++) {

    if( ( (ph*TILE_WIDTH + threadIdx.x) < weight_wid) && (row < M))
    {
     Mds[threadIdx.y][threadIdx.x] = weight[row*weight_wid + ph*TILE_WIDTH + threadIdx.x];
    }else{
      Mds[threadIdx.y][threadIdx.x] = 0;
    }

    if( ( (ph*TILE_WIDTH + threadIdx.y) < K*K*C ) && (col< input_wid)){
     Nds[threadIdx.y][threadIdx.x] = input[(ph*TILE_WIDTH + threadIdx.y)*input_wid + col ];
    }
    else{
      Nds[threadIdx.y][threadIdx.x] = 0;
    }

    __syncthreads();

     for(int i = 0 ; i < TILE_WIDTH ; i ++){
        Pvalue += Mds[threadIdx.y][i] * Nds[i][threadIdx.x];
     }

    __syncthreads();
  }

  if((row < M) && (col < input_wid))
  {
    y4d(b,m,h,w)  = Pvalue;
  }
  /*
  if ((col < input_wid) && (row < M)) {
    for (int i = 0; i< weight_wid; i++) {
        temp += weight[row * weight_wid + i] * input[i * input_wid + col];
    }
    y4d(b,m,h,w) = temp;
  }
*/
  #undef y4d
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
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    // milestone 3 code
    /*
    int X = ceil((M*W_out)/(TILE_WIDTH*1.0));
    int Y = ceil((B*H_out)/(TILE_WIDTH*1.0));
    dim3 gridDim(X, Y, 1);
    dim3 blockDim(TILE_WIDTH,TILE_WIDTH,1);
    forward_kernel<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);
    */

    //milestone 4 code
    float *unrolled_w;
    float *unrolled_x;
    cudaMalloc((void **) &unrolled_w, K*K*C*M * sizeof(float));
    cudaMalloc((void **) &unrolled_x, H*W*B*K*K*C * sizeof(float));

    //unrolling weight
    int X = ceil((C*K*K)/(TILE_WIDTH*1.0));
    int Y = ceil(M/(TILE_WIDTH*1.0));
    dim3 gridDim(X, Y, 1);
    dim3 blockDim(TILE_WIDTH,TILE_WIDTH,1);
    unroll_weight<<<gridDim, blockDim>>>(unrolled_w, w.dptr_, M, C, K);
    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

    //unrolling input
    X = ceil((H_out*W_out*B)/(TILE_WIDTH*1.0));
    Y = ceil((K*K*C)/(TILE_WIDTH*1.0));
    dim3 gridDim2(X, Y, 1);
    dim3 blockDim2(TILE_WIDTH,TILE_WIDTH,1);
    unroll_input<<<gridDim2, blockDim2>>>(unrolled_x, x.dptr_, B, C, H, W, K);
    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

    //matrx mul.
    X = ceil((H_out*W_out*B)/(TILE_WIDTH*1.0));
    Y = ceil(M/(TILE_WIDTH*1.0));
    dim3 gridDim3(X, Y, 1);
    dim3 blockDim3(TILE_WIDTH,TILE_WIDTH,1);
    unroll_multipy<<<gridDim3, blockDim3>>>(unrolled_w, unrolled_x, y.dptr_, B, M, C, H, W, K);
    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

    //free cuda memory
    cudaFree(unrolled_w);
    cudaFree(unrolled_x);

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
