import mindspore
import numpy as np
import Utils.*

def PCANet_FeaExt(InImg, V, X_g_v, X_g_b, TSKNet):
    if len(TSKNet['NumFilters']) != TSKNet['NumStages']:
        print('Length(PCANet.NumFilters) != PCANet.NumStages')
        return None, None

    NumImg = len(InImg)
    OutImg = InImg
    ImgIdx = mindspore.ops.arange(NumImg)
    InImg = None

    for stage in range(TSKNet['NumStages']):
        OutImg, ImgIdx = PCA_output(OutImg, ImgIdx, TSKNet['NumFilters'][stage], TSKNet['PatchSize'][stage], V[stage], X_g_v[stage], X_g_b[stage])

    f, BlkIdx = HashingHist(TSKNet, ImgIdx, OutImg)

    return f, BlkIdx

def extract_patches(img, PatchSize):
    patches = []

    for i in range(img.shape[0] - PatchSize[0] + 1):
        for j in range(img.shape[1] - PatchSize[1] + 1):
            patch = img[i:i+PatchSize[0], j:j+PatchSize[1]]
            patches.append(patch)

    return np.array(patches)



def extract_hist_blocks(img, HistBlockSize, BlkOverLapRatio):
    blocks = []

    for i in range(0, img.shape[0], HistBlockSize[0]):
        for j in range(0, img.shape[1], HistBlockSize[1]):
            block = img[i:i+HistBlockSize[0], j:j+HistBlockSize[1]]
            blocks.append(block)

    return np.array(blocks)