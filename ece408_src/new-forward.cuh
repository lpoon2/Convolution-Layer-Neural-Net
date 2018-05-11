#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_
#include <mxnet/base.h>
__constant__ float weights[2400];

namespace mxnet
{
namespace op
{
__global__ void forward_kernel(float * __restrict__ y, const float * __restrict__ x, const int B, const int M, const int C, const int H, const int W, const int K){
    float sum = 0.0;
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int TILE_WIDTH = (C == 1) ? 16 : 13;
    int X_width = (TILE_WIDTH+K-1);
    extern __shared__ float shmem[];
    float* X_share = &shmem[0];
    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define weights4d(i3, i2, i1, i0) weights[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int W_grid = ceil(W_out/(TILE_WIDTH*1.0));
    int H_grid=ceil(H_out/(TILE_WIDTH*1.0));

    int b = blockIdx.x;
    int m = blockIdx.y;
    int h0 = threadIdx.y;
    int w0 = threadIdx.x;
    int h_base = (blockIdx.z / W_grid)*TILE_WIDTH;
    int w_base = (blockIdx.z % W_grid)*TILE_WIDTH;
    int h = h_base+h0;
    int w = w_base+w0;
    int c = threadIdx.z;


      for(int i=h; i< h_base + X_width ;i+=TILE_WIDTH){
        for(int j=w; j< w_base + X_width;j+=TILE_WIDTH){
          X_share[c*(X_width*X_width)+(i-h_base)*X_width+(j-w_base)]=x4d(b,c,i,j); //load in tile needed for shared memory
        }
      }
      __syncthreads();
      int r0 = (c*(X_width*X_width)+(threadIdx.y)*X_width);
      int r1 = (c*(X_width*X_width)+(threadIdx.y+1)*X_width);
      int r2 = (c*(X_width*X_width)+(threadIdx.y+2)*X_width);
      int r3 = (c*(X_width*X_width)+(threadIdx.y+3)*X_width);
      int r4 = (c*(X_width*X_width)+(threadIdx.y+4)*X_width);
      sum+=(X_share[r0+(threadIdx.x)]*weights4d(m,c,0,0)
      + X_share[r0+(threadIdx.x+1)]*weights4d(m,c,0,1)
      + X_share[r0+(threadIdx.x+2)]*weights4d(m,c,0,2)
      + X_share[r0+(threadIdx.x+3)]*weights4d(m,c,0,3)
      + X_share[r0+(threadIdx.x+4)]*weights4d(m,c,0,4)
      + X_share[r1+(threadIdx.x)]*weights4d(m,c,1,0)
      + X_share[r1+(threadIdx.x+1)]*weights4d(m,c,1,1)
      + X_share[r1+(threadIdx.x+2)]*weights4d(m,c,1,2)
      + X_share[r1+(threadIdx.x+3)]*weights4d(m,c,1,3)
      + X_share[r1+(threadIdx.x+4)]*weights4d(m,c,1,4)
      + X_share[r2+(threadIdx.x)]*weights4d(m,c,2,0)
      + X_share[r2+(threadIdx.x+1)]*weights4d(m,c,2,1)
      + X_share[r2+(threadIdx.x+2)]*weights4d(m,c,2,2)
      + X_share[r2+(threadIdx.x+3)]*weights4d(m,c,2,3)
      + X_share[r2+(threadIdx.x+4)]*weights4d(m,c,2,4)
      + X_share[r3+(threadIdx.x)]*weights4d(m,c,3,0)
      + X_share[r3+(threadIdx.x+1)]*weights4d(m,c,3,1)
      + X_share[r3+(threadIdx.x+2)]*weights4d(m,c,3,2)
      + X_share[r3+(threadIdx.x+3)]*weights4d(m,c,3,3)
      + X_share[r3+(threadIdx.x+4)]*weights4d(m,c,3,4)
      + X_share[r4+(threadIdx.x)]*weights4d(m,c,4,0)
      + X_share[r4+(threadIdx.x+1)]*weights4d(m,c,4,1)
      + X_share[r4+(threadIdx.x+2)]*weights4d(m,c,4,2)
      + X_share[r4+(threadIdx.x+3)]*weights4d(m,c,4,3)
      + X_share[r4+(threadIdx.x+4)]*weights4d(m,c,4,4));
      __syncthreads();

  if ((b < B) && (m < M) && (h < H_out) && (w < W_out)) {
    atomicAdd(&y4d(b,m,h,w), sum);
  }

    #undef y4d
    #undef x4d
    #undef k4d
}
__global__ void forward_kernel2(float * __restrict__ y, const float * __restrict__ x, const int B, const int M, const int C, const int H, const int W, const int K){

    float sum=0.0;
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int TILE_WIDTH = (C == 1) ? 16 : 13;
    int X_width = (TILE_WIDTH+K-1);
    extern __shared__ float shmem[];
    float* X_share = &shmem[0];
    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define weights4d(i3, i2, i1, i0) weights[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int W_grid = ceil(W_out/(TILE_WIDTH*1.0));
    int H_grid=ceil(H_out/(TILE_WIDTH*1.0));

    int b = blockIdx.x;
    int m = blockIdx.y;
    int h0 = threadIdx.y;
    int w0 = threadIdx.x;
    int h_base = (blockIdx.z / W_grid)*TILE_WIDTH;
    int w_base = (blockIdx.z % W_grid)*TILE_WIDTH;
    int h = h_base+h0;
    int w = w_base+w0;

    for(int c=0;c<C;c++){
      for(int i=h; i< h_base + X_width ;i+=TILE_WIDTH){
        int row = (i-h_base)*X_width;
        for(int j=w; j< w_base + X_width;j+=TILE_WIDTH){
          X_share[row+(j-w_base)]=x4d(b,c,i,j); //load in tile needed for shared memory
        }
      }
      __syncthreads();
      int r0 = ((threadIdx.y)*X_width);
      int r1 = ((threadIdx.y+1)*X_width);
      int r2 = ((threadIdx.y+2)*X_width);
      int r3 = ((threadIdx.y+3)*X_width);
      int r4 = ((threadIdx.y+4)*X_width);
      sum+=(X_share[r0+(threadIdx.x)]*weights4d(m,c,0,0)
          + X_share[r0+(threadIdx.x+1)]*weights4d(m,c,0,1)
          + X_share[r0+(threadIdx.x+2)]*weights4d(m,c,0,2)
          + X_share[r0+(threadIdx.x+3)]*weights4d(m,c,0,3)
          + X_share[r0+(threadIdx.x+4)]*weights4d(m,c,0,4)
          + X_share[r1+(threadIdx.x)]*weights4d(m,c,1,0)
          + X_share[r1+(threadIdx.x+1)]*weights4d(m,c,1,1)
          + X_share[r1+(threadIdx.x+2)]*weights4d(m,c,1,2)
          + X_share[r1+(threadIdx.x+3)]*weights4d(m,c,1,3)
          + X_share[r1+(threadIdx.x+4)]*weights4d(m,c,1,4)
          + X_share[r2+(threadIdx.x)]*weights4d(m,c,2,0)
          + X_share[r2+(threadIdx.x+1)]*weights4d(m,c,2,1)
          + X_share[r2+(threadIdx.x+2)]*weights4d(m,c,2,2)
          + X_share[r2+(threadIdx.x+3)]*weights4d(m,c,2,3)
          + X_share[r2+(threadIdx.x+4)]*weights4d(m,c,2,4)
          + X_share[r3+(threadIdx.x)]*weights4d(m,c,3,0)
          + X_share[r3+(threadIdx.x+1)]*weights4d(m,c,3,1)
          + X_share[r3+(threadIdx.x+2)]*weights4d(m,c,3,2)
          + X_share[r3+(threadIdx.x+3)]*weights4d(m,c,3,3)
          + X_share[r3+(threadIdx.x+4)]*weights4d(m,c,3,4)
          + X_share[r4+(threadIdx.x)]*weights4d(m,c,4,0)
          + X_share[r4+(threadIdx.x+1)]*weights4d(m,c,4,1)
          + X_share[r4+(threadIdx.x+2)]*weights4d(m,c,4,2)
          + X_share[r4+(threadIdx.x+3)]*weights4d(m,c,4,3)
          + X_share[r4+(threadIdx.x+4)]*weights4d(m,c,4,4));
      __syncthreads();
  }

  if ( (b < B) && (m < M) && (h < H_out) && (w < W_out)){
    y4d(b,m,h,w)=sum;
  }

    #undef y4d
    #undef x4d
    #undef k4d
}

template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> & __restrict__ y, const mshadow::Tensor<gpu, 4, float> & __restrict__ x, const mshadow::Tensor<gpu, 4, float> & __restrict__ w)
{
    const int B = x.shape_[0]; //batches
    const int M = y.shape_[1]; //output channels
    const int C = x.shape_[1]; //input channels
    const int H = x.shape_[2]; //height of input
    const int W = x.shape_[3]; //width of input
    const int K = w.shape_[3]; //height and width of weights
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int TILE_WIDTH = (C == 1) ? 16 : 13;
    //cudaStream_t s = y.stream_->stream_;
    const int W_grid = ceil(W_out/(1.0*TILE_WIDTH));
    const int H_grid = ceil(H_out/(1.0*TILE_WIDTH));
    const int Z = W_grid*H_grid;

    cudaMemcpyToSymbol(weights, w.dptr_, (K*K*C*M) * sizeof(float));
    //printf("HELLO======= B :%d, M: %d, C: %d, H:%d, W:%d, K:%d", B, M, C, H, W, K);
    if (C == 1){
    dim3 gridDim(B, M, Z);
    dim3 blockDim(TILE_WIDTH,TILE_WIDTH,C);
    size_t shmem_size=sizeof(float)*(((TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1)*C));
    forward_kernel<<<gridDim, blockDim,shmem_size>>>(y.dptr_, x.dptr_, B, M, C, H, W, K);
  } else {
    dim3 gridDim(B, M, Z);
    dim3 blockDim(TILE_WIDTH,TILE_WIDTH,1);
    size_t shmem_size=sizeof(float)*(((TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1)));
    forward_kernel2<<<gridDim, blockDim,shmem_size>>>(y.dptr_, x.dptr_, B, M, C, H, W, K);

  }
    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}

template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif
