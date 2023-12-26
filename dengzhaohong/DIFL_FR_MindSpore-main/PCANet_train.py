import mindspore
import Utils.*

def PCANet_train(InImg, TSKNet, IdtExt):
    if len(TSKNet.NumFilters) != TSKNet.NumStages:
        print('Length(TSKNet.NumFilters) != TSKNet.NumStages')
        return
    
    V = [None] * TSKNet.NumStages
    X_g_v = [None] * TSKNet.NumStages
    X_g_b = [None] * TSKNet.NumStages
    
    NumImg = len(InImg)
    OutImg = InImg
    ImgIdx = mindspore.ops.arange(1, NumImg+1)
    InImg = None
    
    for stage in range(1, TSKNet.NumStages+1):
        print(f'Generating the antecedents parameters of TSK FS by FCM at stage {stage}...')
        X_g_v[stage-1], X_g_b[stage-1] = gene_v_b(OutImg, TSKNet, stage)
        print(f'Computing filter bank and Transforming feature space from X to Xg at stage {stage}...')
        V[stage-1] = PCA_FilterBank(OutImg, TSKNet.PatchSize[stage-1], TSKNet.NumFilters[stage-1],
                                    TSKNet.NumRules[stage-1], X_g_v[stage-1], X_g_b[stage-1])
        
        if stage != TSKNet.NumStages:
            OutImg, ImgIdx = PCA_output(OutImg, ImgIdx, TSKNet.NumFilters[stage-1], TSKNet.PatchSize[stage-1],
                                        V[stage-1], X_g_v[stage-1], X_g_b[stage-1])
    
    if IdtExt == 1:
        f = []
        for idx in range(NumImg):
            if idx % 100 == 0:
                print(f'Extracting PCANet feature of the {idx}th training sample...')
            OutImgIndex = ImgIdx == idx + 1
            OutImg_i, ImgIdx_i = PCA_output(OutImg[OutImgIndex], mindspore.ops.ones(mindspore.ops.sum(OutImgIndex)),
                                            TSKNet.NumFilters[-1], TSKNet.PatchSize[-1],
                                            V[-1], X_g_v[-1], X_g_b[-1])
            f_i, BlkIdx = HashingHist(TSKNet, ImgIdx_i, OutImg_i)
            f.append(f_i)
            OutImg[OutImgIndex] = None
        
        f = mindspore.ops.sparse.cat(f, dim=1)
    else:
        f = None
        BlkIdx = None
    
    return f, V, X_g_v, X_g_b, BlkIdx