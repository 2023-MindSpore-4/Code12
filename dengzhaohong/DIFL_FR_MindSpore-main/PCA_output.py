import mindspore
import Utils.*

def PCA_output(InImg, ImgIdx, NumFilters, PatchSize, V, X_g_v, X_g_b):
    OutImg = []

    for i in range(len(InImg)):
        img = InImg[i]
        img_patches = extract_patches(img, PatchSize)
        img_patches = img_patches.reshape(img_patches.shape[0], -1)

        img_patches = img_patches - X_g_b
        img_patches = img_patches / X_g_v

        img_patches = mindspore.Tensor.from_numpy(img_patches).float().to(device)
        V = mindspore.Tensor.from_numpy(V).float().to(device)

        img_out = F.conv2d(img_patches.unsqueeze(0).unsqueeze(0), V.unsqueeze(1))
        img_out = img_out.squeeze().cpu().numpy()

        OutImg.append(img_out)

    OutImg = np.array(OutImg)
    ImgIdx = ImgIdx.numpy()

    return OutImg, ImgIdx