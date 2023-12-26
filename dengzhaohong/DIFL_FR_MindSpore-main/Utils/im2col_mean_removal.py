import numpy as np

def im2colstep(img, patchsize12, step=1):
    img_height, img_width = img.shape
    patch_height, patch_width = patchsize12

    patches = []
    for i in range(0, img_height - patch_height + 1, step):
        for j in range(0, img_width - patch_width + 1, step):
            patch = img[i:i + patch_height, j:j + patch_width]
            patches.append(patch.flatten())

    return np.array(patches)

def im2col_mean_removal(*args):
    NumInput = len(args)
    InImg = args[0]
    patchsize12 = args[1]

    z = InImg.shape[2]
    im = []
    if NumInput == 2:
        for i in range(z):
            iim = im2colstep(InImg[:, :, i], patchsize12)
            iim_mean = iim - np.mean(iim, axis=0)
            im_i = iim_mean.T
            im.append(im_i)
    else:
        for i in range(z):
            iim = im2colstep(InImg[:, :, i], patchsize12, args[2])
            iim_mean = iim - np.mean(iim, axis=0)
            im_i = iim_mean.T
            im.append(im_i)

    im = np.concatenate(im, axis=0)
    return im


