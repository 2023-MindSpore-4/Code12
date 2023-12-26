from mindspore import Tensor, ops
from conf import conf
from skimage import segmentation, color

from joblib import Parallel, delayed

def __proc(img:Tensor)->Tensor:
    
    img=img.transpose(1,2,0)
    img=Tensor.asnumpy(img)
    seg_img=segmentation.felzenszwalb(img,1,0.8,200)
    out_img=color.label2rgb(seg_img,img,kind="avg")
    out_img=Tensor(out_img)
    out_img=out_img.transpose(2,0,1)
    return out_img

def effecient_segmentation(input:Tensor,*args,**kwargs)->Tensor:
    


    # imgs = [x[i,:,:,:] for i in range(x.shape[0])]

    # # batch_out = Parallel(n_jobs=x.shape[0])(delayed(proc)(img) for img in imgs)

    # batch_out = [proc(img) for img in imgs]

    num_job = input.shape[0]
    batch_out = Parallel(n_jobs=num_job)(delayed(__proc)(image)
                                         for image in input)

    return Tensor(batch_out)