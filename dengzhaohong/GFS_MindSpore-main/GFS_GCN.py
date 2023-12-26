import os
import time
import warnings
from sched import scheduler

from loguru import logger

from Model.GCN_GraphClassifier import GraphClassifier
from Utils.Preprocess import tensor_from_numpy, Preprocess_GandMUandY
from Utils.TrainHelper import mini_batches

warnings.filterwarnings('ignore')
import mindspore
from scipy.special import softmax
from sklearn import preprocessing
from sklearn.metrics import precision_score, accuracy_score, roc_auc_score
import numpy as np
import scipy
from mindspore import nn
from mindspore.train import Model, CheckpointConfig, ModelCheckpoint, LossMonitor


class GraphFuzzySystem(nn.Cell):
    def __int__(self, _X_train: list, _Y_train: list, _X_val: list, _Y_val: list, _X_test: list, _Y_test: list,
                _num_rules: int, _centers: list,
                _num_class: int, mu4Train_list,
                mu4Val_list, mu4Test_list,
                DatasetName, HyperParams: dict,
                mini_batch_size):
        super.__init__()

        self.mini_batch_size = mini_batch_size
        self.X_train = _X_train
        self.Y_train = _Y_train
        self.X_val = _X_val
        self.Y_val = _Y_val
        self.X_test = _X_test
        self.Y_test = _Y_test
        self.num_rules = _num_rules
        self.centers = _centers
        self.num_class = _num_class
        self.graph_process_units = None
        self.DatasetName = DatasetName

        self.mu4train_list = mu4Train_list
        self.mu4val_list = mu4Val_list
        self.mu4test_list = mu4Test_list

        self.NodeFeatures_Train = list()
        self.AdjMat_Train = list()

        self.NodeFeatures_Val = list()
        self.AdjMat_Val = list()

        self.NodeFeatures_Test = list()
        self.AdjMat_Test = list()

        # HyperParameters Settings
        self.EPOCHS = HyperParams['EPOCHS']
        self.HIDDEN_DIM = HyperParams['HIDDEN_DIM']
        self.LEARNING_RATE = HyperParams['LEARNING_RATE']
        self.WEIGHT_DECAY = HyperParams['WEIGHT_DECAY']
        self.PATIENCE = HyperParams['PATIENCE']

        # define loss function and optimizer
        self.criterion = None
        self.optimizer = None

        # for visual
        self.acc_list = list()
        self.loss_list = list()

        # define results store
        self.train_loss_list = list()
        self.train_acc_list = list()
        self.train_macro_f1_list = list()
        self.train_micro_f1_list = list()
        self.train_acc_sk_list = list()
        self.train_auc_roc_list = list()

        self.val_loss_list = list()
        self.val_acc_list = list()
        self.val_macro_f1_list = list()
        self.val_micro_f1_list = list()
        self.val_acc_sk_list = list()
        self.val_auc_roc_list = list()

        self.test_loss_list = list()
        self.test_acc_list = list()
        self.test_macro_f1_list = list()
        self.test_micro_f1_list = list()
        self.test_acc_sk_list = list()
        self.test_auc_roc_list = list()

        self.test_acc = 0
        self.test_macro_f1 = 0
        self.test_micro_f1 = 0
        self.test_acc_sk = 0
        self.test_auc_roc = 0

        # Early Stopping Model Saving Dir
        Time = int(round(time.time() * 1000))
        TimeStr = time.strftime('%Y%m%d_%H%M%S', time.localtime(Time / 1000))
        saved_path = '../TempModel/' + self.DatasetName
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)
        self.saved_path = '../TempModel/{}/{}_GFS-GCN_{}_{}.pkl'.format(self.DatasetName, self.DatasetName, self.DEVICE,
                                                                        TimeStr)

        def fit(self):
            X_4MB = self.X_train
            Mu_4MB = self.mu4train_list
            Y_4MB = self.Y_train

            self.mu4train_list = softmax(self.mu4train_list, axis=1)
            self.mu4train_list = tensor_from_numpy(self.mu4train_list, self.DEVICE)
            seed = 0
            minibatches, num_batch = mini_batches(X_4MB, Mu_4MB, Y_4MB, self.mini_batch_size, seed)

            Orign_X_Val = self.X_val
            Orign_mu4val_list = self.mu4val_list
            Orign_Y_val = self.Y_val
            InputAdjHat_val, Ind_val, y_val, input_dim, X_all_val, MuList_Val = Preprocess_GandMUandY(Orign_X_Val,
                                                                                                      Orign_mu4val_list,
                                                                                                      Orign_Y_val)
            # Create module list
            self.ModuleList = nn.CellList()
            for k in range(self.num_rules):
                self.ModuleList.append(
                    GraphClassifier(input_dim=input_dim, hidden_dim=self.HIDDEN_DIM, num_classes=self.num_class).to(
                        self.DEVICE))

            # Cross entropy loss function
            self.criterion = nn.CrossEntropyLoss().to(self.DEVICE)

            self.optimizer = nn.optim.Adam(params=self.ModuleList.parameters(), lr=self.LEARNING_RATE,
                                           weight_decay=self.WEIGHT_DECAY)
            # scheduler = nn.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)

            logger.info(self.ModuleList)

            # Build  models according to the number of rules
            self.mu4Val_Model = []
            for i in range(MuList_Val.size(1)):
                self.mu4Val_Model.append(MuList_Val[:, i].reshape(MuList_Val.size(0), 1))

            # Initialize the early_stopping object
            best_val_acc, best_val_macro_f1, best_val_micro_f1, count, best_position = 0, 0, 0, 0, 0
            Epoch = list()
            for epoch in range(self.EPOCHS):
                # For plot acc figure
                Epoch.append(epoch + 1)

                # For Train with mini batch
                Count_NumBatch = 0
                self.batch_train_accs = list()
                self.batch_train_macro_f1s = list()
                self.batch_train_micro_f1s = list()
                self.batch_train_roc_aucs = list()

                self.batch_train_losses = list()
                self.batch_train_acc_mean = 0
                self.batch_train_macro_f1_mean = 0
                self.batch_train_micro_f1_mean = 0
                self.batch_train_roc_auc_mean = 0
                self.batch_train_loss_mean = 0

                self.batch_val_accs = list()
                self.batch_val_macro_f1s = list()
                self.batch_val_micro_f1s = list()
                self.batch_val_roc_aucs = list()

                self.batch_val_losses = list()
                self.batch_val_acc_mean = 0
                self.batch_val_macro_f1_mean = 0
                self.batch_val_micro_f1_mean = 0
                self.batch_val_roc_auc_mean = 0
                self.batch_val_loss_mean = 0

                # Training with mini-batch
                for minibatch in minibatches:
                    Count_NumBatch += 1
                    (minibatch_X, minibatch_Mu, minibatch_Y) = minibatch
                    InputAdjHat, Ind, y_train, input_dim, X_all, MuListTrain = Preprocess_GandMUandY(minibatch_X,
                                                                                                     minibatch_Mu,
                                                                                                     minibatch_Y)
                    X_all = nn.tensor(X_all, dtype=nn.float32).to(self.DEVICE)
                    y_train = nn.LongTensor(y_train).to(self.DEVICE)

                    self.ModuleList.train()
                    self.optimizer.zero_grad()

                    self.mu4Train_Model = []
                    logits = nn.Tensor()
                    for i in range(MuListTrain.size(1)):
                        self.mu4Train_Model.append(MuListTrain[:, i].reshape(MuListTrain.size(0), 1))
                        temp_train = self.mu4Train_Model[i] * self.ModuleList[i](InputAdjHat, X_all, Ind, y_train)
                        if i == 0:
                            logits = temp_train
                        else:
                            logits += temp_train

                    loss = self.criterion(logits.to(self.DEVICE), y_train)
                    loss.requires_grad_(True)
                    loss.backward()
                    self.optimizer.step()

                    # Classification metrics
                    train_acc = accuracy_score(y_train.cpu().detach().numpy(),
                                               logits.max(1)[1].cpu().detach().numpy())
                    train_macro_f1_score = precision_score(y_train.cpu().detach().numpy(),
                                                           logits.max(1)[1].cpu().detach().numpy(), average='macro')
                    train_micro_f1_score = precision_score(y_train.cpu().detach().numpy(),
                                                           logits.max(1)[1].cpu().detach().numpy(), average='micro')
                    if self.num_class == 2:
                        train_roc_auc = roc_auc_score(y_train.cpu().detach().numpy(),
                                                      logits.max(1)[1].cpu().detach().numpy())
                    else:
                        train_roc_auc = 0.0
                    self.batch_train_losses.append(loss.item())
                    self.batch_train_accs.append(train_acc)
                    self.batch_train_macro_f1s.append(train_macro_f1_score)
                    self.batch_train_micro_f1s.append(train_micro_f1_score)
                    self.batch_train_roc_aucs.append(train_roc_auc)

                    # For Val
                    logits_val = nn.Tensor()
                    for i in range(MuList_Val.size(1)):
                        temp_val = self.mu4Val_Model[i] * self.ModuleList[i](InputAdjHat_val, nn.tensor(X_all_val,
                                                                                                        dtype=nn.float32).to(
                            self.DEVICE), Ind_val, y_val)
                        if i == 0:
                            logits_val = temp_val
                        else:
                            logits_val += temp_val

                    loss_val = self.criterion(logits_val.to(self.DEVICE), nn.LongTensor(y_val).to(self.DEVICE))
                    val_acc = accuracy_score(y_val, logits_val.max(1)[1].cpu().detach().numpy())
                    val_macro_f1_score = precision_score(y_val, logits_val.max(1)[1].cpu().detach().numpy(),
                                                         average='macro')
                    val_micro_f1_score = precision_score(y_val, logits_val.max(1)[1].cpu().detach().numpy(),
                                                         average='micro')
                    if self.num_class == 2:
                        val_roc_auc = roc_auc_score(y_val, logits_val.max(1)[1].cpu().detach().numpy())
                    else:
                        val_roc_auc = 0.0

                    self.batch_val_losses.append(loss_val.item())
                    self.batch_val_accs.append(val_acc)
                    self.batch_val_macro_f1s.append(val_macro_f1_score)
                    self.batch_val_micro_f1s.append(val_micro_f1_score)
                    self.batch_val_roc_aucs.append(val_roc_auc)
                    logger.info(
                        'Epoch:{:05d}/{:05d}--Batch:{:03d}/{:03d}|Train Loss:{:.6f},Train Acc: {:.4},Train F1-macro: {:.4},Train F1-micro: {:.4},Train ROC-AUC: {:.4}|Val Loss:{:.6f},Val Acc:{:.4},Val F1-macro:{:.4},Val F1-micro:{:.4},Val ROC-AUC:{:.4}'.format(
                            epoch + 1, self.EPOCHS, Count_NumBatch, num_batch, loss.item(), train_acc,
                            train_macro_f1_score, train_micro_f1_score, train_roc_auc,
                            loss_val.item(), val_acc, val_macro_f1_score, val_micro_f1_score, val_roc_auc))

                    if Count_NumBatch == num_batch:
                        self.batch_train_acc_mean = np.mean(self.batch_train_accs)
                        self.batch_train_macro_f1_mean = np.mean(self.batch_train_macro_f1s)
                        self.batch_train_micro_f1_mean = np.mean(self.batch_train_micro_f1s)
                        self.batch_train_roc_auc_mean = np.mean(self.batch_train_roc_aucs)
                        self.batch_train_loss_mean = np.mean(self.batch_train_losses)

                        self.batch_val_acc_mean = np.mean(self.batch_val_accs)
                        self.batch_val_macro_f1_mean = np.mean(self.batch_val_macro_f1s)
                        self.batch_val_micro_f1_mean = np.mean(self.batch_val_micro_f1s)
                        self.batch_val_roc_auc_mean = np.mean(self.batch_val_roc_aucs)
                        self.batch_val_loss_mean = np.mean(self.batch_val_losses)

                        self.train_acc_list.append(self.batch_train_acc_mean)
                        self.train_macro_f1_list.append(self.batch_train_macro_f1_mean)
                        self.train_micro_f1_list.append(self.batch_train_micro_f1_mean)
                        self.train_auc_roc_list.append(self.batch_train_roc_auc_mean)
                        self.train_loss_list.append(self.batch_train_loss_mean)

                        self.val_acc_list.append(self.batch_val_acc_mean)
                        self.val_macro_f1_list.append(self.batch_val_macro_f1_mean)
                        self.val_micro_f1_list.append(self.batch_val_micro_f1_mean)
                        self.val_auc_roc_list.append(self.batch_val_roc_auc_mean)
                        self.val_loss_list.append(self.batch_val_loss_mean)

                scheduler.step()

                logger.info('Current Learning Rate: {}'.format(scheduler.get_last_lr()))
                # Early stopping
                if self.batch_val_macro_f1_mean > best_val_macro_f1 and self.batch_train_micro_f1_mean > best_val_micro_f1:
                    best_val_macro_f1 = self.batch_val_macro_f1_mean
                    best_val_micro_f1 = self.batch_val_micro_f1_mean
                    count = 0
                    best_position = epoch + 1
                    ModelDict = {}
                    for k in range(self.num_rules):
                        ModelDict['Model{}'.format(k)] = self.ModuleList[k].state_dict()

                    nn.save(ModelDict, self.saved_path)
                    logger.info('Saving The Temp Best Val_Acc Model...')
                else:
                    count += 1
                    patience = self.PATIENCE
                    if count > patience:
                        logger.info('Early Stopping Epoch:{}'.format(epoch + 1))

                    break
