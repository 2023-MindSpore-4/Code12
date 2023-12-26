import os
import time
import numpy as np
import pandas
from loguru import logger
from sklearn.model_selection import StratifiedKFold, train_test_split
import pandas as pd
import json
# Private Lib
from Model.GFS_GCN import GraphFuzzySystem
from Utils.GetAntecedent import GetAntWithVal
from Utils.ReadData import read_data

if __name__ == '__main__':

    HyperParams_json = open('./Config/hyperparams.json')
    HyperParamConfig = json.load(HyperParams_json)

    # datasets: PROTEINS ENZYMES BZR COX2 DHFR PROTEINS_full AIDS Cuneiform
    DatasetName = "PROTEINS_full"
    DataContent = read_data(DatasetName, prefer_attr_nodes=True)
    G, y = DataContent.data, DataContent.target
    num_class = len(np.unique(y))

    # Logging
    Time = int(round(time.time() * 1000))
    TimeStr = time.strftime('%Y-%m-%d_%H%M%S', time.localtime(Time / 1000))
    logger.add(
        "./results/logs/{}/GFS_GCN_Data set-{}_TimeStamp-{}.log".format(DatasetName, DatasetName, TimeStr))

    # K-Fold CV
    num_folds = 5
    kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=0)

    Epochs = HyperParamConfig['Epochs']
    best_result = dict()
    best_result['Epochs'] = Epochs
    best_result['best_acc'] = 0
    best_result['best_acc_std'] = 0
    best_result['best_macro_f1'] = 0
    best_result['best_macro_f1_std'] = 0
    best_result['best_micro_f1'] = 0
    best_result['best_micro_f1_std'] = 0

    best_result['best_roc_auc'] = 0
    best_result['best_roc_auc_std'] = 0
    best_result['num_rules'] = 0
    best_result['l2'] = 0
    best_result['lr'] = 0
    best_result['HiddenDim'] = 0
    best_result['Mini_Batch_Size'] = 0

    all_result = list()

    Rules = HyperParamConfig['Rules']
    for rule in Rules:
        temp_res_list = dict()
        num_rules = rule
        temp_res_list['Epochs'] = Epochs
        temp_res_list['num_rules'] = num_rules
        L2s = HyperParamConfig['L2s']  # , 10 ** -3, 10 ** -2, 10 ** -1
        for l2 in L2s:
            temp_res_list['l2'] = l2
            Lrs = HyperParamConfig['Lrs']   # 1e-5, 1e-4, 1e-3, 1e-2, 1e-1
            for lr in Lrs:
                temp_res_list['lr'] = lr
                HiddenDims = HyperParamConfig['HiddenDims']
                for HiddenDim in HiddenDims:
                    temp_res_list['HiddenDim'] = HiddenDim
                    MiniBatchSize = HyperParamConfig['MiniBatchSize']
                    for mini_batch_size in MiniBatchSize:
                        temp_res_list['mini_batch_size'] = mini_batch_size
                        test_accs = list()
                        test_macro_f1_s = list()
                        test_micro_f1_s = list()
                        test_acc_sk_s = list()
                        test_auc_roc_s = list()
                        num_kf = 0
                        for train_index, test_index in kf.split(G, y):
                            num_kf += 1
                            logger.info('Dataset:{}'.format(DatasetName))
                            logger.info('{}-Fold Cross Validation: {}/{}'.format(num_folds, num_kf, num_folds))

                            G_train, y_train = np.array(G)[train_index], y[train_index]
                            G_test, y_test = np.array(G)[test_index], y[test_index]
                            G_train_tra, G_train_val, y_train_tra, y_train_val = train_test_split(G_train, y_train,
                                                                                                  test_size=0.25,
                                                                                                  shuffle=True)
                            # HyperParams:
                            HyperParams = dict()
                            HyperParams['RULES'] = num_rules
                            HyperParams['EPOCHS'] = Epochs
                            HyperParams['HIDDEN_DIM'] = HiddenDim
                            HyperParams['LEARNING_RATE'] = lr
                            HyperParams['PATIENCE'] = HyperParamConfig['Patience']
                            HyperParams['WEIGHT_DECAY'] = l2
                            HyperParams['Mini_Batch_Size'] = mini_batch_size
                            logger.info('HyperParam Settings: {}'.format(HyperParams))

                            # Get the prototype graph
                            ClusterCenters, Mu4Train_list, Mu4Val_list, Mu4Test_list = GetAntWithVal(
                                G_train_tra.tolist(), G_train_val.tolist(), G_test, num_rules)

                            # Build the GraphFuzzySystem model
                            GfsModel = GraphFuzzySystem(_X_train=G_train_tra, _Y_train=y_train_tra,
                                                        _X_val=G_train_val, _Y_val=y_train_val,
                                                        _X_test=G_test, _Y_test=y_test,
                                                        _num_rules=num_rules, _centers=ClusterCenters,
                                                        _num_class=num_class, mu4Train_list=Mu4Train_list,
                                                        mu4Val_list=Mu4Val_list, mu4Test_list=Mu4Test_list,
                                                        DatasetName=DatasetName, HyperParams=HyperParams,
                                                        mini_batch_size=mini_batch_size)
                            GfsModel.fit()
                            GfsModel.predict()
                            # test_acc = GfsModel.return_res()
                            test_acc, test_macro_f1, test_micro_f1, test_auc_roc = GfsModel.return_res()
                            logger.info('Testing Params:{}'.format(HyperParams))
                            logger.info('Testing Acc:{}'.format(test_acc))
                            logger.info('Testing Macro F1:{}'.format(test_macro_f1))
                            logger.info('Testing Micro F1:{}'.format(test_micro_f1))

                            logger.info('Testing AUC ROC:{}'.format(test_auc_roc))

                            test_accs.append(test_acc)
                            test_macro_f1_s.append(test_macro_f1)
                            test_micro_f1_s.append(test_micro_f1)

                            test_auc_roc_s.append(test_auc_roc)
                        test_acc_mean = np.mean(test_accs)
                        test_acc_std = np.std(test_accs)

                        test_macro_f1_mean = np.mean(test_macro_f1_s)
                        test_macro_f1_std = np.std(test_macro_f1_s)

                        test_micro_f1_mean = np.mean(test_micro_f1_s)
                        test_micro_f1_std = np.std(test_micro_f1_s)

                        test_auc_roc_mean = np.mean(test_auc_roc_s)
                        test_auc_roc_std = np.std(test_auc_roc_s)

                        if test_macro_f1_mean > best_result['best_macro_f1'] and test_micro_f1_mean > best_result['best_micro_f1']:
                            best_result['best_acc'] = test_acc_mean
                            best_result['best_acc_std'] = test_acc_std
                            best_result['best_macro_f1'] = test_macro_f1_mean
                            best_result['best_macro_f1_std'] = test_macro_f1_std
                            best_result['best_micro_f1'] = test_micro_f1_mean
                            best_result['best_micro_f1_std'] = test_micro_f1_std
                            best_result['best_roc_auc'] = test_auc_roc_mean
                            best_result['best_roc_auc_std'] = test_auc_roc_std
                            best_result['num_rules'] = num_rules
                            best_result['l2'] = l2
                            best_result['lr'] = lr
                            best_result['HiddenDim'] = HiddenDim
                            best_result['Mini_Batch_Size'] = mini_batch_size
                            # logger.info('The temp best_result:{}'.format(best_result))
                        temp_res_list['test_acc_mean'] = test_acc_mean
                        temp_res_list['test_acc_std'] = test_acc_std
                        temp_res_list['test_macro_f1_mean'] = test_macro_f1_mean
                        temp_res_list['test_macro_f1_std'] = test_macro_f1_std
                        temp_res_list['test_micro_f1_mean'] = test_micro_f1_mean
                        temp_res_list['test_micro_f1_std'] = test_micro_f1_std

                        temp_res_list['test_auc_roc_mean'] = test_auc_roc_mean
                        temp_res_list['test_auc_roc_std'] = test_auc_roc_std

                        # Store Results into CSV
                        file_name = os.path.basename(__file__)
                        file_name = file_name.split('.')[0]
                        # Check the saved path
                        saved_dir = './results/csv/{}'.format(DatasetName)
                        if not os.path.exists(saved_dir):
                            os.makedirs(saved_dir)
                        pandas.DataFrame([best_result]).to_csv('{}/{}_{}.csv'.format(saved_dir, file_name, TimeStr),
                                                               index=False, mode='a')

                        logger.info(best_result)
                        all_result.append(best_result)
    logger.info('All Results:')
    logger.info(all_result)
    logger.info('The Last Best Result:')
    logger.info(best_result)
    logger.info('**********End**************')
