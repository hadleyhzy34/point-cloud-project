
#include "gpu_frnn.hpp"
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
int const threadsPerBlock = 1024;
int const xPerBlock = 32;
int const yPerBlock = 32;

__device__ inline float cuComputeDistance(float const x_0,
                                          float const y_0,
                                          float const z_0,
                                          float const x_1,
                                          float const y_1,
                                          float const z_1){
    float x = (x_0-x_1)*(x_0-x_1);
    float y = (y_0-y_1)*(y_0-y_1);
    float z = (z_0-z_1)*(z_0-z_1);
    /*printf("current returned distance is: %f\n", sqrtf(x+y+z));*/
    return sqrtf(x+y+z);
}

__global__ void _frnn_kernel(const int num_points,
                            const float *points,
                            const int point_dim, 
                            const float radius,
                            int *mask){
    const int row_id_block = blockIdx.y * blockDim.y + threadIdx.y;
    const int col_id_block = blockIdx.x * blockDim.x + threadIdx.x;

    
    int i=0,j=0,row_id=0,col_id=0;
    for(j=0; j*maxThreadsY + row_id_block < num_points; j++){
        row_id = j*maxThreadsY + row_id_block;
        for(i=0; i*maxThreadsX + col_id_block < num_points; i++){
            col_id = i*maxThreadsX + col_id_block;
            float temp_distance = cuComputeDistance(points[row_id*point_dim],
                                                    points[row_id*point_dim+1],
                                                    points[row_id*point_dim+2],
                                                    points[col_id*point_dim],
                                                    points[col_id*point_dim+1],
                                                    points[col_id*point_dim+2]);
            if(temp_distance < radius){
                mask[row_id*num_points+col_id] = 1;
                mask[col_id*num_points+row_id] = 1;
            }
        }
    }
    //printf("last status: i: %d,j: %d,row_id: %d, col_id: %d\n",i,j,row_id,col_id);
    
    //printf("num_points: %d, point_dim: %d, row_id: %d, col_id: %d, points: %f",num_points,point_dim,row_id,col_id,points[row_id*point_dim]);
    /*printf("threadidx:%d, threadidy: %d, blockidx: %d, blockidy: %d, blockDim.y: %d, blockDim.x: %d,radius: %f\n", 
            threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, radius);*/
    /*
    if(row_id < num_points && col_id < num_points){
        float temp_distance = cuComputeDistance(points[row_id*point_dim],
                                                points[row_id*point_dim+1],
                                                points[row_id*point_dim+2],
                                                points[col_id*point_dim],
                                                points[col_id*point_dim+1],
                                                points[col_id*point_dim+2]);*/

        /*printf("num_points: %d, point_dim: %d, row_id: %d, col_id: %d, radius: %f, temp_distance: %f\n",
                num_points,point_dim,row_id,col_id,radius,temp_distance);*/
}


__global__ void frnn_kernel(const int num_points,
                            const float *points,
                            const int point_dim, 
                            const float radius,
                            int *mask){
    const int row_id = blockIdx.y * blockDim.y + threadIdx.y;
    const int col_id = blockIdx.x * blockDim.x + threadIdx.x;

    //printf("num_points: %d, point_dim: %d, row_id: %d, col_id: %d, points: %f",num_points,point_dim,row_id,col_id,points[row_id*point_dim]);
    /*printf("threadidx:%d, threadidy: %d, blockidx: %d, blockidy: %d, blockDim.y: %d, blockDim.x: %d,radius: %f\n", 
            threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, radius);*/

    if(row_id < num_points && col_id < num_points){
        float temp_distance = cuComputeDistance(points[row_id*point_dim],
                                                points[row_id*point_dim+1],
                                                points[row_id*point_dim+2],
                                                points[col_id*point_dim],
                                                points[col_id*point_dim+1],
                                                points[col_id*point_dim+2]);

        /*printf("num_points: %d, point_dim: %d, row_id: %d, col_id: %d, radius: %f, temp_distance: %f\n",
                num_points,point_dim,row_id,col_id,radius,temp_distance);*/

        //mask[row_id*num_points+col_id] = 0;
        //mask[col_id*num_points+row_id] = 0;
        if(temp_distance < radius){
            mask[row_id*num_points+col_id] = 1;
            mask[col_id*num_points+row_id] = 1;
            /*printf("-----num_points: %d, point_dim: %d, row_id: %d, col_id: %d, points: %f-----\n",
                    num_points,point_dim,row_id,col_id,points[row_id*point_dim]);*/
        }
    }
}

void frnn(int num_points,int point_dim, const float* points_host,
          const float radius,int *mask_host){
    float* points_dev = NULL;
    int* mask_dev = NULL;
    //printf("current radius before cuda is: %f", radius);
    CUDA_CHECK(cudaMalloc(&points_dev,num_points*point_dim*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(points_dev,
                          points_host,
                          num_points*point_dim* sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&mask_dev,num_points*num_points*sizeof(int)));
    
    /*
    dim3 blocks(DIVUP(num_points, threadsPerBlock),
                DIVUP(num_points, threadsPerBlock));
    dim3 threads(xPerBlock,yPerBlock);
    */
    dim3 blocks(maxThreadsX/xPerBlock,maxThreadsY/yPerBlock);
    dim3 threads(xPerBlock,yPerBlock);

    _frnn_kernel<<<blocks, threads>>>(num_points,
                                     points_dev,
                                     point_dim,
                                     radius,
                                     mask_dev);

    /*frnn_kernel<<<blocks, threads>>>(num_points,
                                     points_dev,
                                     point_dim,
                                     radius,
                                     mask_dev);*/
    CUDA_CHECK(cudaMemcpy(mask_host,
                          mask_dev,
                          num_points*num_points*sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(points_dev));
    CUDA_CHECK(cudaFree(mask_dev));
}
