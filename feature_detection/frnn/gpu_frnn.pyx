import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "gpu_frnn.hpp":
    void frnn(int,int,np.float32_t*,float,np.int32_t*)
    # void _nms(np.int32_t*, int*, np.float32_t*, int, int, float, int)

def gpu_frnn(np.ndarray[np.float32_t, ndim=2] points, np.float radius):
    cdef int num_points = points.shape[0]
    cdef int point_dim = points.shape[1]
    # cdef float radius = r
    cdef np.ndarray[np.int32_t,ndim=1] mask = \
            np.zeros(num_points*num_points,dtype=np.int32)
    cdef np.ndarray[np.float32_t,ndim=1] points_dev = \
            points.reshape((num_points*3))
    # print(f'points:{points},radius:{radius}')
    frnn(num_points,point_dim,&points_dev[0],radius,&mask[0])
    return mask
