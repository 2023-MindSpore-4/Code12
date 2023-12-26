import mindspore
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
import Utils.*

def DeepTSK_demo(DeepTSK, ImgFormat, TrnData, TrnLabels, TestData, TestLabels):
    nTestImg = len(TestLabels)

    print('\n ====== DeepTSK Parameters ======= \n')
    print(DeepTSK)

    print('\n ====== DeepTSK Training ======= \n')
    ImgSize = DeepTSK['ImgSize']
    TrnData_ImgCell = mat2imgcell(TrnData, ImgSize, ImgFormat)
    TrnData = None
    TSKNet_TrnTime = time.time()
    ftrain, V, X_g_v, X_g_b, BlkIdx = PCANet_train(TrnData_ImgCell, DeepTSK)
    TSKNet_TrnTime = time.time() - TSKNet_TrnTime
    TrnData_ImgCell = None

    print('\n ====== Training Linear SVM Classifier ======= \n')
    LinearSVM_TrnTime = time.time()
    models = svm.LinearSVC(C=1)
    models.fit(np.transpose(ftrain), TrnLabels)
    LinearSVM_TrnTime = time.time() - LinearSVM_TrnTime
    ftrain = None

    TestData_ImgCell = mat2imgcell(TestData, ImgSize, ImgFormat)
    TestData = None

    print('\n ====== DeepTSK Testing ======= \n')

    nCorrRecog = 0
    RecHistory = np.zeros(nTestImg)

    test_time = time.time()
    for idx in range(nTestImg):
        ftest = PCANet_FeaExt(TestData_ImgCell[idx], V, X_g_v, X_g_b, DeepTSK)

        xLabel_est = models.predict(np.transpose(ftest))
        accuracy = accuracy_score([TestLabels[idx]], xLabel_est)

        if xLabel_est == TestLabels[idx]:
            RecHistory[idx] = 1
            nCorrRecog += 1

        if idx % (nTestImg // 100) == 0:
            print('Accuracy up to {} tests is {:.2f}%; taking {:.2f} secs per testing sample on average.'.format(
                idx, 100 * nCorrRecog / idx, (time.time() - test_time) / idx))

        TestData_ImgCell[idx] = None

    TestData_ImgCell = None
    Averaged_TimeperTest = (time.time() - test_time) / nTestImg
    Accuracy = nCorrRecog / nTestImg

    print('\n ===== Results of DeepTSK, followed by a linear SVM classifier =====')
    print('     TSKNet training time: {:.2f} secs.'.format(TSKNet_TrnTime))
    print('     Linear SVM training time: {:.2f} secs.'.format(LinearSVM_TrnTime))
    print('     Testing Accuracy rate: {:.2f}%'.format(100 * Accuracy))
    print('     Average testing time {:.2f} secs per test sample.\n'.format(Averaged_TimeperTest))

    with open('log.txt', 'a') as fid:
        fid.write('\n ===== Results of DeepTSK, followed by a linear SVM classifier =====')
        fid.write('\n     TSKNet training time: {:.2f} secs.'.format(TSKNet_TrnTime))
        fid.write('\n     Linear SVM training time: {:.2f} secs.'.format(LinearSVM_TrnTime))
        fid.write('\n     Testing Accuracy rate: {:.2f}%'.format(100 * Accuracy))
        fid.write('\n     Average testing time {:.2f} secs per test sample. \n\n'.format(Averaged_TimeperTest))

    return Accuracy, models, Averaged_TimeperTest, TSKNet_TrnTime, LinearSVM_TrnTime, DeepTSK
