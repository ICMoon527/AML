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
import ppscore as pps
from matplotlib import pyplot as plt

AUC_dic = {}
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

    return accuracy, auc


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
    parser.add_argument('--test_model_path', default='Data/Models/Transformer_Lion_91.45_checkpoint.t7', type=str)
    parser.add_argument('-shuffle', '--shuffle', action='store_true')
    parser.add_argument("--max_length", type=int, default=10000)

    args = parser.parse_args()

    SomeUtils.try_make_dir(args.save_dir)
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.shuffle = False
    args.deterministic = True

    """
    Read Data
    """
    # discard_protein_ID_list = [5, 6, 7, 11, 13, 14]
    # discard_protein_ID_list = [5, 6, 10, 12]
    if args.dataset == 'Data/DataInPatientsUmap':
        trainset = AMLDataset.AMLDataset(args, True)
        testset = AMLDataset.AMLDataset(args, False)
        if args.deterministic:
            trainloader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=16, worker_init_fn=np.random.seed(1234))
            testloader = DataLoader(testset, batch_size=args.batchsize, shuffle=False, num_workers=16, worker_init_fn=np.random.seed(1234))
        else:
            trainloader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=16)
            testloader = DataLoader(testset, batch_size=args.batchsize, shuffle=False, num_workers=16)
    else:
        discard_protein_ID_list = []
        trainset = AMLDataset.AMLDataset(args, True, setZeroClassNum=discard_protein_ID_list)
        testset = AMLDataset.AMLDataset(args, False, setZeroClassNum=discard_protein_ID_list)
        if args.deterministic:
            trainloader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=16, worker_init_fn=np.random.seed(1234))
            testloader = DataLoader(testset, batch_size=args.batchsize, shuffle=False, num_workers=16, worker_init_fn=np.random.seed(1234))
        else:
            trainloader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=16)
            testloader = DataLoader(testset, batch_size=args.batchsize, shuffle=False, num_workers=16)

    """
    Choose model
    """
    # 单独测试
    file = torch.load(args.test_model_path)
    model, epoch, accuracy = file['model'], file['epoch'], file['accuracy']

    model.to(args.device)

    if args.device == torch.device('cuda') and torch.cuda.device_count()>1:
        model = torch.nn.DataParallel(model, range(torch.cuda.device_count()))  # 并行
    cudnn.benchmark = True  # 统一输入大小的情况下能加快训练速度


###############################################################################################
    # set up a logger
    logger = setup_logger(args.model+'_'+args.optimizer, args.save_dir, 0, args.model+'_'+args.optimizer+'_testlog.txt', mode='w+')
    best_result = 100

    """
    TEST
    """
    logger.info('='*20+'Testing Model'+'='*20)
    new_best, auc = test(best_result, args, model, epoch, testloader, logger, discard_protein_name='All reserved', color_num=3)

    # # 去掉一个测试对结果的影响
    # ablation_dic = dict()
    # protein_list = ['SSC-A', 'FSC-A', 'FSC-H', 'CD7', 'CD11B', 'CD13', 'CD19', 'CD33', 'CD34', 'CD38', 'CD45', 'CD56', 'CD117', 'DR', 'HLA-DR']
    # # protein_list = ['SSC-A', 'FSC-A', 'FSC-H', 'CD7', 'CD11B', 'CD13', 'CD33', 'CD34', 'CD38', 'CD45', 'CD56', 'CD117', 'HLA-DR']
    # for i in range(len(protein_list)):
    #     if protein_list[i] in 'DR':
    #         continue
    #     if i not in discard_protein_ID_list:
    #         if i == 14:
    #             testset = AMLDataset.AMLDataset(args, False, [i-1, i])
    #         else:
    #             testset = AMLDataset.AMLDataset(args, False, [i])
    #         testloader = DataLoader(testset, batch_size=args.batchsize, shuffle=True, num_workers=16, worker_init_fn=np.random.seed(1234))
    #         new_best, auc = test(best_result, args, model, epoch, testloader, logger, discard_protein_name=protein_list[i], color_num=i)  # 6.3589s
    #         ablation_dic[protein_list[i]] = [new_best, auc]
    # print(ablation_dic)
    # np.save('Results/Test/AUC_dic.npy', AUC_dic)

    # ablation_dic_001 = {'SSC-A': [62.83185840707964, 0.8620053655264923], 'FSC-A': [80.53097345132744, 0.9057679409792085], 'FSC-H': [60.176991150442475, 0.9076123407109322], 'CD7': [62.83185840707964, 0.8253688799463448], 'CD11B': [78.76106194690266, 0.9618544600938967], 'CD13': [82.30088495575221, 0.9858316566063046], 'CD19': [81.85840707964601, 0.9936284372904092], 'CD33': [81.85840707964601, 0.9891851106639838], 'CD34': [42.0353982300885, 0.9897719651240777], 'CD38': [89.38053097345133, 0.9771965124077799], 'CD45': [83.1858407079646, 0.9746814218645203], 'CD56': [95.13274336283186, 0.9866700201207242], 'CD117': [71.68141592920354, 0.9539738430583502], 'HLA-DR': [91.15044247787611, 0.9315057008718981]}
    # ablation_dic_002 = {'SSC-A': [48.95104895104895, 0.9585127201565559], 'FSC-A': [65.03496503496504, 0.9906066536203523], 'FSC-H': [65.03496503496504, 0.9886497064579256], 'CD7': [79.02097902097903, 0.997651663405088], 'CD11B': [92.3076923076923, 0.9818003913894324], 'CD13': [90.9090909090909, 0.9927592954990214], 'CD33': [50.34965034965035, 0.9473581213307241], 'CD34': [62.93706293706294, 0.6796477495107632], 'CD38': [54.54545454545455, 0.3273972602739726], 'CD45': [65.03496503496504, 0.585518590998043], 'CD56': [100.0, 1.0], 'CD117': [91.60839160839161, 0.9972602739726028], 'HLA-DR': [86.01398601398601, 0.9937377690802348]}



    # 剪除蛋白组
    # if '002' in args.save_dir:
    #     pass
    # else:  # 001
    #     protein_list = ['SSC-A', 'FSC-A', 'FSC-H', 'CD7', 'CD11B', 'CD13', 'CD19', 'CD33', 'CD34', 'CD38', 'CD45', 'CD56', 'CD117', 'DR', 'HLA-DR']
    #     discard_protein_ID_list = [5, 6, 7, 11, 13, 14]
    #     testset = AMLDataset.AMLDataset(args, False, discard_protein_ID_list)
    #     testloader = DataLoader(testset, batch_size=args.batchsize, shuffle=True, num_workers=16, worker_init_fn=np.random.seed(1234))
    #     new_best = test(best_result, args, model, epoch, testloader, logger)
###############################################################################################