
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

__constant__ float weight[2400];

__global__ void unroll_weight(float *output_w, const float *k, const int M, const int C, const int K) {
  int m = (blockIdx.y * blockDim.y + threadIdx.y);
  int c = (blockIdx.x * blockDim.x + threadIdx.x) / (K*K);
  int h = ((blockIdx.x * blockDim.x + threadIdx.x) % (K*K)) / K;
  int w = (blockIdx.x * blockDim.x + threadIdx.x) % K;
  int idx = (C*K*K) * (blockIdx.y * blockDim.y + threadIdx.y) + (blockIdx.x * blockDim.x + threadIdx.x);
  #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

  if (((blockIdx.x * blockDim.x + threadIdx.x) < (K*K*C)) && ((blockIdx.y * blockDim.y + threadIdx.y) < M)) {
    output_w[idx] = k4d(m,c,h,w);
  }
  #undef k4d
}

__global__ void unroll_multipy(float *input, const float *x, float *y, const int B, const int M, const int C, const int H, const int W, const int K) {
  const int H_out = H - K + 1;
  const int W_out = W - K + 1;
  #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
  #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]

  int start_h = ((blockIdx.x * blockDim.x + threadIdx.x) % (H_out*W_out)) / W_out;
  int start_w = ((blockIdx.x * blockDim.x + threadIdx.x) % (H_out*W_out)) % W_out;
  int h = start_h + (((blockIdx.y * blockDim.y + threadIdx.y) % (K*K)) / K);
  int w = start_w + (((blockIdx.y * blockDim.y + threadIdx.y) % (K*K)) % K);
  int c = (blockIdx.y * blockDim.y + threadIdx.y) / (K*K);
  int b = (blockIdx.x * blockDim.x + threadIdx.x) / (H_out*W_out);
  int idx = (H_out*W_out*B) * (blockIdx.y * blockDim.y + threadIdx.y) + (blockIdx.x * blockDim.x + threadIdx.x);

  if (((blockIdx.x * blockDim.x + threadIdx.x) < (H_out*W_out*B)) && ((blockIdx.y * blockDim.y + threadIdx.y) < (K*K*C))) {
    input[idx] = x4d(b,c,h,w);
  }
  __syncthreads();

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int weight_wid = K*K*C;
  int input_wid = H_out*W_out*B;
  b = col / (H_out*W_out);
  int m = row;
  h = (col % (H_out*W_out)) / W_out;
  w = (col % (H_out*W_out)) % W_out;

  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
  float Pvalue = 0.0 ;

  for(int ph = 0 ; ph < (weight_wid-1)/TILE_WIDTH + 1 ; ph++) {

    if( ( (ph*TILE_WIDTH + threadIdx.y) < K*K*C ) && (col< input_wid)){
      Nds[threadIdx.y][threadIdx.x] = input[(ph*TILE_WIDTH + threadIdx.y)*input_wid + col ];
    } else {
      Nds[threadIdx.y][threadIdx.x] = 0;
    }

    __syncthreads();

     for(int i = 0 ; i < TILE_WIDTH ; i ++){
        Pvalue += weight[row*weight_wid + ph*TILE_WIDTH + i]* Nds[i][threadIdx.x];
     }

    __syncthreads();
  }

  if((row < M) && (col < input_wid))
  {
    y4d(b,m,h,w)  = Pvalue;
  }


  #undef x4d
  #undef y4d
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

    //milestone 4 code
    float *unrolled_w;
    float *unrolled_x;
	  float *host_unrolled_w;

	  cudaMalloc((void **) &unrolled_w, K*K*C*M * sizeof(float));
	  host_unrolled_w = (float*)malloc(2400 * sizeof(float));
    cudaMalloc((void **) &unrolled_x, H*W*B*K*K*C * sizeof(float));

    //unrolling weight
    int X = ceil((C*K*K)/(TILE_WIDTH*1.0));
    int Y = ceil(M/(TILE_WIDTH*1.0));
    dim3 gridDim(X, Y, 1);
    dim3 blockDim(TILE_WIDTH,TILE_WIDTH,1);
    unroll_weight<<<gridDim, blockDim>>>(unrolled_w, w.dptr_, M, C, K);
    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

  	//transferring the unrolled weight to the host
  	cudaMemcpy(host_unrolled_w, unrolled_w, K*K*C*M*sizeof(float), cudaMemcpyDeviceToHost);
  	cudaMemcpyToSymbol(weight, host_unrolled_w, K*K*C*M*sizeof(float));

    //matrx mul.
    X = ceil((H_out*W_out*B)/(TILE_WIDTH*1.0));
    Y = ceil((K*K*C)/(TILE_WIDTH*1.0));
    dim3 gridDim3(X, Y, 1);
    dim3 blockDim3(TILE_WIDTH,TILE_WIDTH,1);
    unroll_multipy<<<gridDim3, blockDim3>>>(unrolled_x, x.dptr_, y.dptr_, B, M, C, H, W, K);
    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

    //free cuda memory
    cudaFree(unrolled_w);
    cudaFree(unrolled_x);
    free(host_unrolled_w);
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
