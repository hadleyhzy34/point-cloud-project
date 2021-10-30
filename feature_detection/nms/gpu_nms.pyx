import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "gpu_nms.hpp":
    void nms(int,int,np.int32_t*,np.int32_t*,np.int32_t*,np.float32_t*)

def gpu_nms(np.ndarray[np.int32_t,ndim=1] mask,
            np.ndarray[np.int32_t,ndim=1] mask_idx,
            np.ndarray[np.int32_t,ndim=2] adj,
            np.ndarray[np.float32_t,ndim=1] l3):
    cdef int num_points = mask.shape[0]
    cdef int num_keypoints = mask_idx.shape[0]
    cdef np.ndarray[np.int32_t,ndim=1] c_adj = \
            adj.reshape((num_points*num_points))
    cdef np.ndarray[np.int32_t,ndim=1] c_mask = mask
    cdef np.ndarray[np.int32_t,ndim=1] c_mask_idx = mask_idx
    cdef np.ndarray[np.float32_t,ndim=1] c_l3 = l3
    # print(f'points:{points},radius:{radius}')
    print(f'current data list: {num_keypoints},{num_points},{c_adj},{c_mask},{c_mask_idx},{c_l3}')
    nms(num_keypoints,
        num_points,
        &c_adj[0],
        &c_mask[0],
        &c_mask_idx[0],
        &c_l3[0])
    return c_mask
