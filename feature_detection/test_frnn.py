import numpy as np
import time
#from nms.nums_py2 import py_cpu_nms  # for cpu
from frnn.gpu_frnn import gpu_frnn


if __name__ == "__main__":
    import ipdb;ipdb.set_trace()
    points = np.random.randn(10000,3).astype(np.float32)
    mask = gpu_frnn(points,2.0)
    print(mask)



