import mindspore
import Utils.*

def X2Xg(OutImg, PatchSize, v, b):
    img = padding(OutImg, PatchSize)
    im = im2col_general(img, [PatchSize, PatchSize])
    im = mindspore.tensor(im).t()
    im_xg = calc_x_g(im, v, b)
    im_xg = im_xg.t()
    im_xg = im_xg - mindspore.ops.mean(im_xg, dim=0)
    
    return im_xg