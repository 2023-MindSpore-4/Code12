import numpy as np
import Utils.*

def PCA_FilterBank(InImg, PatchSize, NumFilters, NumRules, X_g_v, X_g_b):
    ImgZ = len(InImg)
    MaxSamples = 100000
    NumRSamples = min(ImgZ, MaxSamples)
    RandIdx = np.random.permutation(ImgZ)[:NumRSamples]

    NumChls = InImg[0].shape[2]
    Rx = np.zeros(((NumChls * PatchSize**2 + 1) * NumRules, (NumChls * PatchSize**2 + 1) * NumRules))

    for i in RandIdx:
        im = X2Xg(InImg[i], PatchSize, X_g_v, X_g_b)
        Rx += np.dot(im, im.T)
        InImg[i] = None

    Rx /= (NumRSamples * im.shape[1])
    E, D = np.linalg.eig(Rx)
    ind = np.argsort(np.diag(D))[::-1]
    V = E[:, ind[:NumFilters]]
    return V