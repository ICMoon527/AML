from torch.utils.data import Dataset
from utils.ReadCSV import ReadCSV
from sklearn.model_selection import train_test_split
import numpy as np
import torch

class AMLDataset(Dataset):
    def __init__(self, args, isTrain=True, setZeroClassNum='None') -> None:
        super(AMLDataset, self).__init__()
        self.args = args
        self.isTrain = isTrain

        '''
        读取数据, M2-粒细胞-0, M5-单细胞-1
        '''
        object = ReadCSV()
        X, Y = object.getDataset(args.dataset, length=args.length)
        X, Y = self.preprocess(X, Y)  # (num, 10000, 15)
        # self.MAX = np.max(X)
        
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=np.random.seed(1234))
        print('训练集长度: {}, 测试集长度: {}'.format(len(self.X_train), len(self.X_test)))
        
        if isTrain:
            self.X = self.X_train
            self.Y = self.Y_train
        else:
            self.X = self.X_test
            self.Y = self.Y_test

        '''
        其他数据操作
        '''
        # if isTrain:
        #     self.X, self.Y = self.dataMixing(self.X, self.Y)  # 把训练集的病人流式细胞信息混合
        #     self.X, self.Y = self.dataAugmentation(self.X, self.Y, range_=10, times=2)
        #     self.X, self.Y = self.addPerturbation(self.X, self.Y, 0.1)
        #     self.X = self.X / np.max(self.X)
        # np.random.seed(1234)
        # self.fix_indices = np.random.choice(np.arange(len(self.X[0])), replace=False,
        #                                 size=int(len(self.X[0])*5.1/6.0))
        if not setZeroClassNum == 'None':
            self.X[:, :, setZeroClassNum] = 0
        np.random.seed(1234)

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x_origin = x.copy()
        # if self.args.input_droprate > 0:
        #     """
        #     some detectors fail
        #     """
        #     assert self.args.input_droprate < 1
        #     if self.isTrain==False and self.args.dropout_rate>0 and self.args.input_droprate>0:  # 在测试时不要加双重dropout
        #         indices = np.random.choice(np.arange(len(x)), replace=False,
        #                                 size=int(len(x) * self.args.input_droprate))
        #         x[indices] = 0  # some detectors fail
        #     elif self.args.dropout_rate==0 and self.args.input_droprate>0:  # 训练和测试都让输入随机失活
        #         indices = np.random.choice(np.arange(len(x)), replace=False,
        #                                 size=int(len(x) * self.args.input_droprate))
        #         x[indices] = 0  # some detectors fail

        # random choose 51 sensors to be ZERO
        # x[self.fix_indices] = 0
        return np.float32(x.flatten()), np.int32(y), np.float32(x_origin)  # 让类别从0开始

    def __len__(self):
        return len(self.X)

    def countBigDataNum(self, array, threshold):
        count = np.sum(array>threshold)
        return count

    def dataAugmentation(self, input, target, range_=10, times=2):
        input_charac_num = int(input.shape[-1])
        if self.isTrain:
            # 训练阶段，复制多份数据，并在选定范围内缩放，默认上下浮动20%
            new_X = np.expand_dims(input, axis=0)
            new_Y = np.expand_dims(target, axis=0)
            new_X = np.repeat(new_X, 2*range_+1, axis=0)
            new_Y = np.repeat(new_Y, 2*range_+1, axis=0)
            for i in range(2*range_+1):
                new_X[i] = new_X[i] / 100. * (100+times*(i-range_))  # scale
            new_X = new_X.reshape(-1, self.args.length, input_charac_num)
            new_Y = new_Y.reshape(-1)
            return new_X, new_Y
        else:
            # 在测试阶段，不变
            return input, target

    def preprocess(self, x, y):
        # min-max scale
        x = x / 1023.
        x = x.transpose(0, 2, 1)  # (num, 10000, 15)

        return x, y

    def addPerturbation(self, x, y, scale=0.01):
        if self.isTrain:
            # 训练阶段，对训练数据加入随机百分比微扰
            scale = np.random.random(x.shape) * scale * 2 - scale  # (-scale, scale)
            x = x * (scale+1)
            return x, y
        else:
            # 测试阶段，对测试数据加入随机百分比微扰
            # scale = np.random.random(x.shape) * scale * 2 - scale  # (-scale, scale)
            # x = x * (scale+1)
            return x, y
        
    def dataMixing(self, x, y):
        x_0, x_1 = [], []
        y_0, y_1 = [], []
        for i, item in enumerate(y):
            if item == 0:
                x_0.append(x[i])
                y_0.append(y[i])
            elif item == 1:
                x_1.append(x[i])
                y_1.append(y[i])
            else:
                print('ERROR in dataMixing')
                exit()
        x_0, x_1 = np.array(x_0), np.array(x_1)

        x_0, x_1 = x_0.reshape((x_0.shape[0]*x_0.shape[1], x_0.shape[-1])), x_1.reshape((x_1.shape[0]*x_1.shape[1], x_1.shape[-1]))
        np.random.shuffle(x_0)
        np.random.shuffle(x_1)
        x_0, x_1 = x_0.reshape((int(len(x_0)/10000), 10000, x_0.shape[-1])), x_1.reshape((int(len(x_1)/10000), 10000, x_1.shape[-1]))
        x, y = np.vstack([x_0, x_1]), np.hstack([y_0, y_1])
            
        return x, y

if __name__ == '__main__':
    import argparse
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['SVM', 'DNN', 'ATTDNN', 'preDN', 'DNNATT', 'UDNN', 'Resume'], default='DNN')
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--length", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--optimizer', default='Adam', type=str, choices=['SGD','Adam','Adamax'])
    parser.add_argument('--save_dir', default='./Results/DNN', type=str)
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
    parser.add_argument('--dataset', default='Data/UsefulData', type=str, choices=['Data/UsefulData','Data/UsefulData002'])
    parser.add_argument('-train', '--train', action='store_true')
    parser.add_argument('--test_model_path', default='Results/DNN-notShuffle-dropout0d5/DNN_Adam_98.23_checkpoint.t7', type=str)
    args = parser.parse_args()

    object = AMLDataset(args)
    input, target, _ = object.__getitem__(0)
    print(input.shape)
    # indices = np.random.choice(np.arange(len(input)), replace=False,
    #                        size=int(len(input) * 0.2))
    # input[indices] = 0
    # print(input)
    # print(target)