import os
import random
import argparse
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import Model
import AMLDataset
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.optim as optim
from utils.logger import setup_logger
import time
from utils import SomeUtils, FocalLoss
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split, StratifiedKFold

AUC_dic = {}

def getPatientScaledDataXY(max_length=10000):
    X_train = list()
    patient_cell_num = list()
    Y_train = list()
    Umap_1_max = -10000
    Umap_1_min = 10000
    Umap_2_max = -10000
    Umap_2_min = 10000
    for root, dirs, files in os.walk('Data/DataInPatientsUmap'):
        for file in files:
            if 'npy' in file:
                print('Proceeding {}...'.format(file))
                numpy_data = np.load(os.path.join(root, file))  # shape均值267000

                Umap_1_max = Umap_1_max if Umap_1_max > np.max(numpy_data[:,0]) else np.max(numpy_data[:,0])
                Umap_1_min = Umap_1_min if Umap_1_min < np.min(numpy_data[:,0]) else np.min(numpy_data[:,0])
                Umap_2_max = Umap_2_max if Umap_2_max > np.max(numpy_data[:,1]) else np.max(numpy_data[:,1])
                Umap_2_min = Umap_2_min if Umap_2_min < np.min(numpy_data[:,1]) else np.min(numpy_data[:,1])

                cell_group_num = 0
                # 截长补短
                while numpy_data.shape[0] >= max_length:
                    X_train.append(numpy_data[:max_length])
                    Y_train.append(int(file.split('_')[-1][0]))  # 0:M2, 1:M5
                    numpy_data = numpy_data[max_length:]
                    cell_group_num += 1
                if len(numpy_data) > 0:
                    X_train.append(numpy_data)
                    Y_train.append(int(file.split('_')[-1][0]))  # 0:M2, 1:M5
                    cell_group_num += 1
                
                patient_cell_num.append(cell_group_num)

    # standarize
    for i in range(len(X_train)):
        X_train[i][:,0] = (X_train[i][:,0]-Umap_1_min)/(Umap_1_max-Umap_1_min)
        X_train[i][:,1] = (X_train[i][:,1]-Umap_2_min)/(Umap_2_max-Umap_2_min)
        X_train[i] = torch.tensor(X_train[i])

    X_train = torch.nn.utils.rnn.pad_sequence(X_train, batch_first=True, padding_value=0)
    return np.array(X_train), np.array(Y_train), patient_cell_num

def test(best_result, args, model, epoch, testloader, logger, model_att=None, discard_protein_name=None, color_num=-1):
    model.eval()
    correct = 0.
    total = 0

    predicted_list = []
    target_list = []
    score_list = []

    for batch_idx, (inputs, targets, _) in enumerate(testloader):
        if args.device == torch.device('cuda'):
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, requires_grad=False), Variable(targets)

        if model_att is not None:
            inputs = model_att(inputs)
        out = model(inputs)  # (batch, nClasses)
        _, predicted = torch.max(out.detach(), 1)
        correct += predicted.eq(targets.detach()).sum().item()
        total += targets.size(0)
        score = (out.detach().cpu().numpy())[:, 0]

        predicted_list.append(predicted.cpu().numpy())
        target_list.append(targets.cpu().numpy())
        score_list.append(score)
    
    predicted_list = np.hstack(predicted_list)
    target_list = np.hstack(target_list)
    score_list = np.hstack(score_list)

    classification_report = metrics.classification_report(target_list, predicted_list, target_names=['Patient Type '+str(i+1) for i in range(2)])
    # 计算混淆矩阵
    cm = confusion_matrix(target_list, predicted_list)
    # 可视化
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["M2 - Class 0", "M5 - Class 1"])
    disp.plot(cmap='Blues', values_format='d')  # 'd' 表示显示整数
    plt.title("Confusion Matrix")
    plt.savefig(args.save_dir+'/Confusion_Matrix.png', dpi=600)
    plt.clf()
    
    # AUC(Area Under Curve), ROC(Receiver Operating Characteristic)，正样本为0(M2)
    fig, ax = plt.subplots()
    fpr, tpr, thresholds = metrics.roc_curve(target_list, score_list, pos_label=0)
    AUC_dic[discard_protein_name] = [fpr, tpr]
    auc = metrics.auc(fpr, tpr)
    logger.info('AUC: {}'.format(auc))

    colors = ['pink', 'grey', 'rosybrown', 'red', 'chocolate', 'tan', 'orange', 'lawngreen', 'darkgreen', 'aquamarine', 'dodgerblue', 'blue', 'darkviolet', 'magenta', 'brown', 'black']
    ax.plot(fpr, tpr, label=round(auc, 3), color=colors[color_num])
    ax.legend()
    ax.set_xlabel("FPR", fontweight='bold')
    ax.set_ylabel("TPR", fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
    plt.savefig(args.save_dir+'/AUC.png', dpi=600)

    accuracy = 100. * float(correct) / float(total)
    logger.info("\n| Validation Epoch #%d\t\t\taccuracy =  %.4f" % (epoch, accuracy))
    logger.info('classification report: \n{}'.format(classification_report))
    if accuracy > best_result:
        logger.info('\n| Saving Best model...\t\t\taccuracy = %.4f > %.4f (Best Before)' % (accuracy, best_result))
        state = {
            'model': model.module if isinstance(model, torch.nn.DataParallel) else model,
            'accuracy': accuracy,
            'epoch': epoch,
        }
        
        prefix = args.model + '_' + args.optimizer + '_' + str(accuracy)[0:5] + '_'
        torch.save(state, os.path.join(args.save_dir, prefix+'checkpoint.t7'))
        best_result = accuracy
    else:
        logger.info('\n| Not best... {:.4f} < {:.4f}'.format(accuracy, best_result))

    return accuracy, auc, predicted_list, target_list


class AMLDataset(Dataset):
    def __init__(self, args, isTrain=True, groupThreshold=20) -> None:
        super(AMLDataset, self).__init__()
        self.args = args
        self.isTrain = isTrain

        '''
        读取数据, M2-粒细胞-0, M5-单细胞-1
        '''
        X, Y, patient_cell_num = getPatientScaledDataXY(max_length=args.max_length)  # 对齐，所以max_length调整成70000
        
        patients_num = len(patient_cell_num)
        skip_num = 0
        train_patient_num = patients_num//5*4
        for i in range(train_patient_num):
            skip_num += patient_cell_num[i]

        new_X, new_Y = [], []
        for i in range(patients_num - train_patient_num):
            patient_data_X = X[skip_num: skip_num+patient_cell_num[train_patient_num+i]]
            patient_data_Y = Y[skip_num: skip_num+patient_cell_num[train_patient_num+i]]
            skip_num += len(patient_data_X)
            patient_data_X = patient_data_X[:-1]
            patient_data_Y = patient_data_Y[:-1]
            if patient_cell_num[train_patient_num+i] < groupThreshold:  # 病人细胞数量少则补，并做数据增强  
                while len(patient_data_X) < groupThreshold:
                    # 生成第二维的随机排列索引
                    np.random.seed(len(patient_data_X))
                    shuffled_indices = np.random.permutation(patient_data_X.shape[1])
                    # 打乱第二维
                    shuffled_X = patient_data_X[:, shuffled_indices, :]
                    shuffled_Y = patient_data_Y
                    patient_data_X = np.vstack((patient_data_X, shuffled_X))
                    patient_data_Y = np.hstack((patient_data_Y, shuffled_Y))
            patient_cell_num[train_patient_num+i] = len(patient_data_X)
                
            new_X.append(patient_data_X)
            new_Y.append(patient_data_Y)

        self.X = np.vstack(new_X)  # 422
        self.Y = np.hstack(new_Y)
        np.random.seed(1234)
        self.patient_cell_num = patient_cell_num[train_patient_num:]


    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x_origin = x.copy()
        
        if self.args.model == 'Transformer':
            return np.float32(x), np.float32(y), np.float32(x_origin)
        else:
            return np.float32(x.flatten()), np.int32(y), np.float32(x_origin)

    def __len__(self):
        return len(self.X)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['SVM', 'DNN', 'ATTDNN', 'preDN', 'DNNATT', 'UDNN', 'Resume', 'Transformer'], default='Transformer')
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batchsize", type=int, default=150)
    parser.add_argument("--length", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--optimizer', default='Adam', type=str, choices=['SGD','Adam','Adamax', 'Lion'])
    parser.add_argument('--save_dir', default='Results/TestModelResults', type=str)
    parser.add_argument('--nonlin', default="elu", type=str, choices=["relu", "elu", "softplus", 'sigmoid'])
    parser.add_argument('--weight_decay', default=0., type=float, help='coefficient for weight decay')
    parser.add_argument('-deterministic', '--deterministic', dest='deterministic', action='store_true',
                       help='fix random seeds and set cuda deterministic')
    parser.add_argument('--warmup_steps', default=10, type=int)
    parser.add_argument('--warmup_start_lr', default=1e-5, type=float)
    parser.add_argument('--power', default=0.5, type=float)
    parser.add_argument('-batchnorm', '--batchnorm', action='store_true')
    parser.add_argument('--dropout_rate', default=0.1, type=float)
    parser.add_argument('--nClasses', default=2, type=int)
    parser.add_argument('--input_droprate', default=0., type=float, help='the max rate of the detectors may fail')
    parser.add_argument('--initial_dim', default=256, type=int)
    parser.add_argument('--continueFile', default='./Results/79sources/DNN-Adam-0-3000-largerRange-focalLoss/bk.t7', type=str)
    parser.add_argument('--dataset', default='Data/DataInPatientsUmap', type=str, choices=['Data/UsefulData','Data/UsefulData002', 'Data/DataInPatientsUmap'])
    parser.add_argument('-train', '--train', action='store_true')
    parser.add_argument('--test_model_path', default='Results/TestModelResults/91.45/Transformer_Lion_91.45_checkpoint.t7', type=str)
    parser.add_argument('-shuffle', '--shuffle', action='store_true')
    parser.add_argument("--max_length", type=int, default=10000)

    args = parser.parse_args()

    SomeUtils.try_make_dir(args.save_dir)
    args.device = torch.device('cpu')
    args.shuffle = False
    args.deterministic = True

    """
    Read Data
    """
    # trainset = AMLDataset(args, True)
    testset = AMLDataset(args, False)
    
    # trainloader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=16, worker_init_fn=np.random.seed(1234))
    testloader = DataLoader(testset, batch_size=args.batchsize, shuffle=False, num_workers=16, worker_init_fn=np.random.seed(1234))

    

    """
    Choose model
    """
    # 单独测试
    file = torch.load(args.test_model_path, map_location=args.device)
    model, epoch, accuracy = file['model'], file['epoch'], file['accuracy']
    model.to(args.device)



    ###############################################################################################
    # set up a logger
    logger = setup_logger(args.model+'_'+args.optimizer, args.save_dir, 0, args.model+'_'+args.optimizer+'_testlog.txt', mode='w+')
    best_result = 100

    """
    TEST
    """
    logger.info('='*20+'Testing Model'+'='*20)
    new_best, auc, predicted_list, target_list = test(best_result, args, model, epoch, testloader, logger, discard_protein_name='All reserved', color_num=3)

    patient_cell_num = testset.patient_cell_num
    start = 0
    correct_patient = 0
    score_list = []
    target_patient_list = []
    predicted_list, target_list = np.array(predicted_list), np.array(target_list)
    for i in range(len(patient_cell_num)):
        group_num = patient_cell_num[i]
        correct = np.sum(predicted_list[start: start+group_num] == target_list[start: start+group_num])
        if correct > (group_num//2):
            correct_patient += 1

        if np.unique(target_list[start: start+group_num])[0] == 1:
            score_list.append(float(correct)/float(group_num))
        else:
            score_list.append(1-(float(correct)/float(group_num)))
        target_patient_list.append(np.unique(target_list[start: start+group_num])[0])

        start += group_num

    print('correct patient num: ', correct_patient, 'ratio: ', float(correct_patient)/float(len(patient_cell_num)))

    # AUC(Area Under Curve), ROC(Receiver Operating Characteristic)，正样本为0(M2)
    fig, ax = plt.subplots()
    fpr, tpr, thresholds = metrics.roc_curve(target_patient_list, -np.array(score_list), pos_label=0)
    
    auc = metrics.auc(fpr, tpr)
    logger.info('AUC: {}'.format(auc))

    colors = ['pink', 'grey', 'rosybrown', 'red', 'chocolate', 'tan', 'orange', 'lawngreen', 'darkgreen', 'aquamarine', 'dodgerblue', 'blue', 'darkviolet', 'magenta', 'brown', 'black']
    ax.plot(fpr, tpr, label=round(auc, 3), color=colors[3])
    ax.legend()
    ax.set_xlabel("FPR", fontweight='bold')
    ax.set_ylabel("TPR", fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
    plt.savefig(args.save_dir+'/AUC.png', dpi=900)