import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "gpu_nms.hpp":
    void nms(int,np.int32_t*,np.int32_t*,np.int32_t*,np.float32_t*)
    # void _nms(np.int32_t*, int*, np.float32_t*, int, int, float, int)

def gpu_nms(np.ndarray[np.int32_t,ndim=1] is_keypoint,
            np.ndarray[np.int32_t,ndim=1] keypoint_indices,
            np.ndarray[np.int32_t,ndim=2] adj,
            np.ndarray[np.float32_t,ndim=1] l3):
    cdef int num_points = is_keypoint.shape[0]
    cdef int num_keypoints = keypoint_indices.shape[0]
    cdef np.ndarray[np.int32_t,ndim=1] adj_dev = \
            adj.reshape((num_points,num_points))
    cdef np.ndarray[np.int32_t,ndim=1] keypoint_indices_dev = keypoint_indices
    cdef np.ndarray[np.int32_t,ndim=1] is_keypoint_dev = is_keypoint
    cdef np.ndarray[np.float32_t,ndim=1] l3_dev = l3
    # print(f'points:{points},radius:{radius}')
    nms(num_keypoints,&adj_dev[0],&is_keypoint_dev[0],\
        &keypoint_indices_dev[0],&l3_dev[0])
    return is_keypoint_dev
