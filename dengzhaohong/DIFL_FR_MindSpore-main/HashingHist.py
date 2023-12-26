import mindspore
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import vstack as sparse_vstack
import Utils.*


def Heaviside(X):
    X = mindspore.ops.sign(X)
    X[X <= 0] = 0
    return X

def vec(X):
    return X.view(-1)

def im2col_general(A, block_size, stride):
    unfold = mindspore.nn.Unfold(block_size, stride)
    return unfold(A.unsqueeze(0)).squeeze(0)

def spp(blkwise_fea, sam_coordinate, ImgSize, pyramid):
    dSize, _ = blkwise_fea.shape
    img_width = ImgSize[1]
    img_height = ImgSize[0]
    pyramid_Levels = len(pyramid)
    pyramid_Bins = pyramid ** 2
    tBins = sum(pyramid_Bins)
    beta = mindspore.ops.zeros(dSize, tBins)
    cnt = 0

    for i1 in range(pyramid_Levels):
        Num_Bins = pyramid_Bins[i1]
        wUnit = img_width / pyramid[i1]
        hUnit = img_height / pyramid[i1]
        xBin = mindspore.ops.ceil(sam_coordinate[0] / wUnit).int()
        yBin = mindspore.ops.ceil(sam_coordinate[1] / hUnit).int()
        idxBin = (yBin - 1) * pyramid[i1] + xBin

        for i2 in range(Num_Bins):
            cnt += 1
            sidxBin = mindspore.ops.nonzero(idxBin == i2 + 1).squeeze()
            if len(sidxBin) == 0:
                continue
            beta[:, cnt-1] = blkwise_fea[:, sidxBin].max(axis=1)[0]

    return beta

def HashingHist(PCANet, ImgIdx, OutImg):
    NumImg = mindspore.ops.max(ImgIdx).item()
    f = [None] * NumImg
    map_weights = mindspore.tensor(2 ** (PCANet['NumFilters'][-1] - 1 - mindspore.ops.arange(PCANet['NumFilters'][-1])))

    for Idx in range(1, NumImg+1):
        Idx_span = mindspore.ops.nonzero(ImgIdx == Idx).squeeze()
        NumOs = len(Idx_span) // PCANet['NumFilters'][-1]
        Bhist = [None] * NumOs

        for i in range(NumOs):
            T = mindspore.ops.zeros_like(OutImg[Idx_span[PCANet['NumFilters'][-1] * i]])
            ImgSize = OutImg[Idx_span[PCANet['NumFilters'][-1] * i]].size()

            for j in range(PCANet['NumFilters'][-1]):
                T += map_weights[j] * Heaviside(OutImg[Idx_span[PCANet['NumFilters'][-1] * i + j]])
                OutImg[Idx_span[PCANet['NumFilters'][-1] * i + j]] = None

            if PCANet['HistBlockSize'] is None:
                NumBlk = mindspore.ops.ceil((PCANet['ImgBlkRatio'] - 1) / PCANet['BlkOverLapRatio']) + 1
                HistBlockSize = mindspore.ops.ceil(mindspore.tensor(T.size()) / PCANet['ImgBlkRatio']).int()
                OverLapinPixel = mindspore.ops.ceil((mindspore.tensor(T.size()) - HistBlockSize) / (NumBlk - 1)).int()
                NImgSize = (NumBlk - 1) * OverLapinPixel + HistBlockSize
                Tmp = mindspore.ops.zeros(tuple(NImgSize.tolist()))
                Tmp[:T.size(0), :T.size(1)] = T
                Bhist[i] = csr_matrix(np.histogram(im2col_general(Tmp, HistBlockSize, OverLapinPixel), bins=range(2 ** PCANet['NumFilters'][-1] + 1))[0])
            else:
                stride = mindspore.ops.round((1 - PCANet['BlkOverLapRatio']) * PCANet['HistBlockSize']).int()
                blkwise_fea = csr_matrix(np.histogram(im2col_general(T, PCANet['HistBlockSize'], stride), bins=range(2 ** PCANet['NumFilters'][-1] + 1))[0])

                if PCANet['Pyramid'] is not None:
                    x_start = mindspore.ops.ceil(PCANet['HistBlockSize'][1] / 2).int()
                    y_start = mindspore.ops.ceil(PCANet['HistBlockSize'][0] / 2).int()
                    x_end = mindspore.ops.floor(ImgSize[1] - PCANet['HistBlockSize'][1] / 2).int()
                    y_end = mindspore.ops.floor(ImgSize[0] - PCANet['HistBlockSize'][0] / 2).int()

                    sam_coordinate = mindspore.ops.stack([
                        mindspore.ops.kron(mindspore.ops.arange(x_start, x_end + 1, stride), mindspore.ops.ones_like(mindspore.ops.arange(y_start, y_end + 1, stride))),
                        mindspore.ops.kron(mindspore.ops.ones_like(mindspore.ops.arange(x_start, x_end + 1, stride)), mindspore.ops.arange(y_start, y_end + 1, stride))
                    ], axis=0)

                    blkwise_fea = spp(blkwise_fea, sam_coordinate, ImgSize, PCANet['Pyramid']).T
                else:
                    blkwise_fea = blkwise_fea.multiply(2 ** PCANet['NumFilters'][-1] / blkwise_fea.sum(axis=0))

                Bhist[i] = blkwise_fea

        f[Idx-1] = vec(sparse_vstack(Bhist).T)

        if PCANet['Pyramid'] is not None:
            f[Idx-1] = f[Idx-1] / mindspore.ops.norm(f[Idx-1])

    f = mindspore.ops.cat(f, axis=1)

    if PCANet['Pyramid'] is not None:
        BlkIdx = mindspore.ops.kron(mindspore.ops.arange(Bhist[0].shape[0]), mindspore.ops.ones(len(Bhist) * Bhist[0].shape[1])).long()
    else:
        BlkIdx = mindspore.ops.kron(mindspore.ops.ones(NumOs), mindspore.ops.kron(mindspore.ops.arange(Bhist[0].shape[1]), mindspore.ops.ones(Bhist[0].shape[0]))).long()

    return f, BlkIdx