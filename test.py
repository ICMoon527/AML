from utils.SomeUtils import draw_train_test
import numpy as np
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=['SVM', 'DNN', 'ATTDNN', 'preDN', 'DNNATT', 'UDNN', 'Resume', 'Transformer'], default='Transformer')
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--batchsize", type=int, default=1)
parser.add_argument("--length", type=int, default=1000)
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else 'cpu')
parser.add_argument('--optimizer', default='Adam', type=str, choices=['SGD','Adam','Adamax', 'Lion'])
parser.add_argument('--save_dir', default='./Results/UMAP_Results', type=str)
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
parser.add_argument('--test_model_path', default='Results/DNN-notShuffle-dropout0d4/DNN_Adam_98.23_checkpoint.t7', type=str)
parser.add_argument('-shuffle', '--shuffle', action='store_true')
parser.add_argument("--max_length", type=int, default=10000)
parser.add_argument("--val_delta", type=int, default=1)

args = parser.parse_args()

train = np.load('Results/UMAP_Results/trainloss.npy')
test = np.load('Results/UMAP_Results/testloss.npy')
draw_train_test(args, train.tolist(), test.tolist(), 'test')