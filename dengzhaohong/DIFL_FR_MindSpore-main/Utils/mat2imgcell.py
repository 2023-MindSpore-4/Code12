import numpy as np

def mat2imgcell(D, height, width, ImgFormat):
    N = D.shape[1]
    if ImgFormat == 'gray':
        Img = [None] * N
        for i in range(N):
            Img[i] = np.reshape(D[:, i], (height, width))
    elif ImgFormat == 'color':
        Img = [None] * N
        for i in range(N):
            Img[i] = np.reshape(D[:, i], (height, width, 3))
    elif ImgFormat == 'color2THREEgray':
        Img = [None] * (3 * N)
        for i in range(N):
            T = np.reshape(D[:, i], (height, width, 3))
            Img[(i-1)*3 + 0] = T[:, :, 0]
            Img[(i-1)*3 + 1] = T[:, :, 1]
            Img[(i-1)*3 + 2] = T[:, :, 2]
    
    return Img
