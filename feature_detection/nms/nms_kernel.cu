#include "gpu_nms.hpp"
#include <math.h>
#include <vector>
#include <iostream>

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
//int const threadsPerBlock = sizeof(unsigned long long) * 16;
int const maxThreadsX = 320;
int const maxThreadsY = 320;
int const xPerBlock = 32;
int const yPerBlock = 32;

__global__ void _nms_kernel(const int num_keypoints,
                            const int num_points, 
                            const int *adj,
                            int *mask,
                            const int *mask_idx,
                            const float *l3){
    const int row_id_block = blockIdx.y * blockDim.y + threadIdx.y;
    const int col_id_block = blockIdx.x * blockDim.x + threadIdx.x;
    
    int i=0,j=0,row_id=0,col_id=0;
    for(j=0; j*maxThreadsY + row_id_block < num_keypoints; j++){
        row_id = j*maxThreadsY + row_id_block;
        int kp_idx = mask_idx[row_id];
        for(i=0; i*maxThreadsX + col_id_block < num_keypoints; i++){
            col_id = i*maxThreadsX + col_id_block;
            int kp_idy = mask_idx[col_id];
            if(adj[row_id*num_keypoints+col_id]&&mask[kp_idx]){
                if(l3[kp_idx]<l3[kp_idy])mask[kp_idx]=false;
            }
        }
    }
}

void nms(const int num_keypoints,
         const int num_points,
         const int* adj,
         int* mask,
         const int* mask_idx,
         const float* l3){
    int* mask_dev = NULL;
    int* mask_idx_dev = NULL;
    int *adj_dev = NULL;
    float* l3_dev = NULL;
    
    CUDA_CHECK(cudaMalloc(&mask_dev,num_points*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(mask_dev,
                          mask,
                          num_points*sizeof(int),
                          cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMalloc(&mask_idx_dev,num_keypoints*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(mask_idx_dev,
                          mask_idx,
                          num_keypoints*sizeof(int),
                          cudaMemcpyHostToDevice));
     
    CUDA_CHECK(cudaMalloc(&adj_dev,num_points*num_points*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(adj_dev,
                          adj,
                          num_points*num_points*sizeof(int),
                          cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMalloc(&l3_dev,num_points*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(l3_dev,
                          l3,
                          num_points*sizeof(float),
                          cudaMemcpyHostToDevice));
    
    dim3 blocks(maxThreadsX/xPerBlock,maxThreadsY/yPerBlock);
    dim3 threads(xPerBlock,yPerBlock);

    _nms_kernel<<<blocks, threads>>>(num_keypoints,
                                     num_points,
                                     adj_dev,
                                     mask_dev,
                                     mask_idx_dev,
                                     l3_dev);
    
    CUDA_CHECK(cudaMemcpy(mask,
                          mask_dev,
                          num_points*sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(adj_dev));
    CUDA_CHECK(cudaFree(l3_dev));
    CUDA_CHECK(cudaFree(mask_dev));
    CUDA_CHECK(cudaFree(mask_idx_dev));
}
