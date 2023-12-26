import numpy as np
from sklearn.cluster import KMeans


def padding(img, PatchSize):
    pad_size = PatchSize // 2
    return np.pad(img, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='symmetric')

def im2col_general(img, patch_size):
    img_height, img_width, num_channels = img.shape
    patch_height, patch_width = patch_size

    patches = np.zeros((img_height - patch_height + 1, img_width - patch_width + 1, patch_height * patch_width * num_channels))

    for i in range(patch_height):
        for j in range(patch_width):
            patch_channel = img[i:i + img_height - patch_height + 1, j:j + img_width - patch_width + 1, :]
            patch_channel = patch_channel.reshape(-1, num_channels)
            patches[:, :, i * patch_width + j::patch_height * patch_width] = patch_channel

    return patches.reshape(-1, patch_height * patch_width * num_channels)

def gene_ante_deter(data, NumRules):
    kmeans = KMeans(n_clusters=NumRules)
    kmeans.fit(data)
    v = kmeans.cluster_centers_
    b = np.zeros((NumRules, data.shape[1]))
    for i in range(NumRules):
        v1 = np.tile(v[i], (data.shape[0], 1))
        u = kmeans.labels_ == i
        uu = np.tile(u[:, np.newaxis], (1, data.shape[1]))
        b[i] = np.sum((data - v1) ** 2 * uu, axis=0) / np.sum(uu, axis=0) / 1

    b = b + np.finfo(np.float32).eps
    return v, b


def gene_v_b(InImgCell, PCANet, stage):
    NumChls = InImgCell[0].shape[2]
    PatchSize = PCANet['PatchSize'][stage]
    NumRules = PCANet['NumRules'][stage]
    ImgZ = len(InImgCell)
    Imgsize = InImgCell[0].shape[0]
    MaxSamples = 100000

    gene_v_b_num = 300
    NumRSamples = min(ImgZ, MaxSamples)
    RandIdx = np.random.permutation(NumRSamples)[:gene_v_b_num]

    gene_vb_data = np.zeros((PatchSize * PatchSize * NumChls, gene_v_b_num * Imgsize * Imgsize))

    cnt = 0
    for i in RandIdx:
        img = padding(InImgCell[i], PatchSize)
        data_i = im2col_general(img, (PatchSize, PatchSize))

        gene_vb_data[:, cnt * Imgsize * Imgsize:(cnt + 1) * Imgsize * Imgsize] = data_i
        cnt = cnt + 1

    gene_vb_data = gene_vb_data.T

    v, b = gene_ante_deter(gene_vb_data, NumRules)

    return v, b