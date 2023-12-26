import mindspore
from sklearn.model_selection import KFold
import Utils.*

ImgSize = 32
ImgFormat = 'gray'  # 'color' or 'gray'

trainset = mindspore.dataset.FashionMNIST(root='./data', train=True, download=True)
testset = mindspore.dataset.FashionMNIST(root='./data', train=False, ownload=True)

Datas = trainset.data.view(trainset.data.shape[0], -1).numpy().T
labels = trainset.targets.numpy()
TestData = testset.data.view(testset.data.shape[0], -1).numpy().T
TestLabels = testset.targets.numpy()

k_cv = 5
_, n = Datas.shape
kf = KFold(n_splits=k_cv, shuffle=True, random_state=1234)

rules1 = list(range(2, 16))
rules2 = [3, 5, 7]
NumFilters1 = [8]
NumFilters2 = [8]

best_acc_te = 0
all_result = []
for NumRule1 in rules1:
    for NumRule2 in rules2:
        for NumFilter1 in NumFilters1:
            for NumFilter2 in NumFilters2:
                base_acc_te = 0
                result = []
                for train_index, test_index in kf.split(Datas.T):
                    train_data, test_data = Datas[:, train_index], Datas[:, test_index]
                    train_labels, test_labels = labels[train_index], labels[test_index]

                    DeepTSK = {}
                    DeepTSK['ImgSize'] = ImgSize
                    DeepTSK['NumRules'] = [NumRule1]
                    DeepTSK['NumStages'] = 1
                    DeepTSK['PatchSize'] = [7, 7]
                    DeepTSK['NumFilters'] = [NumFilter1]
                    DeepTSK['HistBlockSize'] = [7, 7]
                    DeepTSK['BlkOverLapRatio'] = 0.5
                    DeepTSK['Pyramid'] = []

                    Accuracy, models, Averaged_TimeperTest, TSKNet_TrnTime, LinearSVM_TrnTime, TSKNet = DeepTSK_demo(
                        DeepTSK, ImgFormat, train_data, train_labels, test_data, test_labels)
                    result.append(Accuracy)

                acc_te_mean = mindspore.ops.mean(mindspore.tensor(result))
                acc_te_std = mindspore.ops.std(mindspore.tensor(result))
                acc_te_min = mindspore.ops.min(mindspore.tensor(result))
                acc_te_max = mindspore.ops.max(mindspore.tensor(result))

                all_result_s = [NumRule1, NumFilter1, acc_te_mean]
                all_result.append(all_result_s)
                if acc_te_mean > best_acc_te:
                    best_result = {}
                    best_result['mean'] = acc_te_mean
                    best_result['std'] = acc_te_std
                    best_result['min'] = acc_te_min
                    best_result['max'] = acc_te_max
                    best_result['model'] = models
                    best_result['Averaged_TimeperTest'] = Averaged_TimeperTest
                    best_result['TSKNet_TrnTime'] = TSKNet_TrnTime
                    best_result['LinearSVM_TrnTime'] = LinearSVM_TrnTime
                    best_result['TSKNet'] = TSKNet
                    mindspore.save(best_result, 'best_result_5CV/best_result_EYaleB_32_2stage_rules.pth')
                    best_acc_te = acc_te_mean

mindspore.save(all_result, 'best_result_5CV/all_result_EYaleB_32_2stage_rules.pth')
