import numpy as np

def padding(InImg, PatchSize):
    mag = (PatchSize - 1) // 2
    ImgX, ImgY, NumChls = InImg.shape
    img = np.zeros((ImgX + PatchSize - 1, ImgY + PatchSize - 1, NumChls))
    img[(mag + 1):(-mag), (mag + 1):(-mag), :] = InImg
    
    return img
