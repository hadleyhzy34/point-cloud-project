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

__global__ void _nms_kernel(int num_keypoints,
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
            if(
            if(mask[row_id]==true&&l3[row_id]<l3[col_id]){
                mask[row_id]=false;
            }
        }
    }
}

void nms(int num_keypoints,int* adj,int* mask,int* mask_idx,const float* l3){
    float* l3_dev = NULL;
    int* mask_dev = NULL;
    //printf("current radius before cuda is: %f", radius);
    CUDA_CHECK(cudaMalloc(&l3_dev,num_points*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(l3_dev,
                          l3,
                          num_points*sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&mask_dev,num_points*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(mask,
                          mask_dev,
                          num_points*sizeof(float),
                          cudaMemcpyHostToDevice));
    
    dim3 blocks(maxThreadsX/xPerBlock,maxThreadsY/yPerBlock);
    dim3 threads(xPerBlock,yPerBlock);

    _nms_kernel<<<blocks, threads>>>(mask_dev,
                                     l3_dev,
                                     num_points);

    CUDA_CHECK(cudaMemcpy(mask,
                          mask_dev,
                          num_points*num_points*sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(l3_dev));
    CUDA_CHECK(cudaFree(mask_dev));
}
