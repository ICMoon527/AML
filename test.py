import os
import random
import argparse
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn import metrics
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


def test(best_result, args, model, epoch, testloader, logger, model_att=None):
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
    
    # AUC(Area Under Curve), ROC(Receiver Operating Characteristic)，正样本为0(M2)
    fpr, tpr, thresholds = metrics.roc_curve(target_list, score_list, pos_label=0)
    auc = metrics.auc(fpr, tpr)
    logger.info('AUC: {}'.format(auc))

    from matplotlib import pyplot as plt
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(args.save_dir+'/AUC.png')

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

    return best_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['SVM', 'DNN', 'ATTDNN', 'preDN', 'DNNATT', 'UDNN', 'Resume'], default='DNN')
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--length", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--optimizer', default='Adam', type=str, choices=['SGD','Adam','Adamax'])
    parser.add_argument('--nonlin', default="elu", type=str, choices=["relu", "elu", "softplus", 'sigmoid'])
    parser.add_argument('--weight_decay', default=0., type=float, help='coefficient for weight decay')
    parser.add_argument('-deterministic', '--deterministic', dest='deterministic', action='store_true',
                       help='fix random seeds and set cuda deterministic')
    parser.add_argument('--warmup_steps', default=10, type=int)
    parser.add_argument('--warmup_start_lr', default=1e-5, type=float)
    parser.add_argument('--power', default=0.5, type=float)
    parser.add_argument('-batchnorm', '--batchnorm', action='store_true')
    parser.add_argument('--dropout_rate', default=0., type=float)
    parser.add_argument('--nClasses', default=2, type=int)
    parser.add_argument('--input_droprate', default=0., type=float, help='the max rate of the detectors may fail')
    parser.add_argument('--initial_dim', default=256, type=int)
    parser.add_argument('--continueFile', default='./Results/79sources/DNN-Adam-0-3000-largerRange-focalLoss/bk.t7', type=str)
    parser.add_argument('-train', '--train', action='store_true')
    
    parser.add_argument('--save_dir', default='./Results/Test', type=str)
    parser.add_argument('--dataset', default='Data/UsefulData', type=str, choices=['Data/UsefulData','Data/UsefulData002'])
    parser.add_argument('--test_model_path', default='Results/DNN-1000epochs-withNorm/DNN_Adam_94.93_checkpoint.t7', type=str)

    args = parser.parse_args()

    SomeUtils.try_make_dir(args.save_dir)
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    """
    Read Data
    """
    trainset = AMLDataset.AMLDataset(args, True)
    testset = AMLDataset.AMLDataset(args, False)
    if args.deterministic:
        trainloader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=16, worker_init_fn=np.random.seed(1234))
        testloader = DataLoader(testset, batch_size=args.batchsize, shuffle=True, num_workers=16, worker_init_fn=np.random.seed(1234))
    else:
        trainloader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=16)
        testloader = DataLoader(testset, batch_size=args.batchsize, shuffle=True, num_workers=16)

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
    logger = setup_logger(args.model+'_'+args.optimizer, args.save_dir, 0, args.model+'_'+args.optimizer+'_log.txt', mode='w+')
    best_result = 100

    """
    TEST
    """
    logger.info('='*20+'Testing Model'+'='*20)
    new_best = test(best_result, args, model, epoch, testloader, logger)

    for i in range(13):
        testset = AMLDataset.AMLDataset(args, False, i)
        testloader = DataLoader(testset, batch_size=args.batchsize, shuffle=True, num_workers=16, worker_init_fn=np.random.seed(1234))
        new_best = test(best_result, args, model, epoch, testloader, logger)
###############################################################################################