from torch.utils.data import Dataset
from utils.ReadCSV import ReadCSV
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from UnsupSajid import getPatientScaledDataXY

class AMLDataset(Dataset):
    def __init__(self, args, isTrain=True, setZeroClassNum=None) -> None:
        super(AMLDataset, self).__init__()
        self.args = args
        self.isTrain = isTrain

        '''
        读取数据, M2-粒细胞-0, M5-单细胞-1
        '''
        if args.dataset == 'Data/DataInPatientsUmap':
            X, Y = getPatientScaledDataXY(max_length=args.max_length)
            self.all_X, self.all_Y = X, Y
        else:
            object = ReadCSV()
            X, Y = object.getDataset(args.dataset, length=args.length)
            print('读取数据完成')
            X, Y = self.preprocess(X, Y)  # (num, 10000, 15)
            print('数据预处理（归一化）完成')
            self.all_X, self.all_Y = X.reshape((-1, X.shape[-1])), Y
        
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.2, shuffle=args.shuffle, random_state=np.random.seed(1234))
        # np.save('Data/npyData/X_train.npy', self.X_train)
        # np.save('Data/npyData/X_test.npy', self.X_test)
        # np.save('Data/npyData/Y_train.npy', self.Y_train)
        # np.save('Data/npyData/Y_test.npy', self.Y_test)
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
        if setZeroClassNum:
            for item in setZeroClassNum:
                self.X[:, :, item] = 0
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
        if not self.args.dataset == 'Data/DataInPatientsUmap':
            return np.float32(x.flatten()), np.int32(y), np.float32(x_origin)  # 让类别从0开始
        else:
            return np.float32(x), np.float32(y), np.float32(x_origin)

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
    
    def getPPScore(self):
        import ppscore as pps
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt

        useful_items = ['SSC-A', 'FSC-A', 'FSC-H', 'CD7', 'CD11B', 'CD13', 'CD19', 'CD33', 'CD34', 'CD38', 'CD45', 'CD56', 'CD117', 'DR', 'HLA-DR']
        # useful_items = ['SSC-A', 'FSC-A', 'FSC-H', 'CD7', 'CD11B', 'CD13', 'CD33', 'CD34', 'CD38', 'CD45', 'CD56', 'CD117', 'HLA-DR']
        # PPSCORE
        df = pd.DataFrame()
        Y = list()
        for i in range(len(self.all_X)):
            if self.all_Y[int(i/10000)] == 1:
                Y.append('True')
            else:
                Y.append('False')
        for i in range(self.all_X.shape[1]):
            df[useful_items[i]] = self.all_X[:, i]
        df["y"] = Y

        df.drop(columns=['DR'], inplace=True)
        # print(df.head())

        predictors_df = pps.predictors(df, y="y")
        print(predictors_df)
        # fig = sns.barplot(data=predictors_df, x="x", y="ppscore")
        # fig.get_figure().savefig('PPSResults/001.png', dpi=400)

        matrix_df = pps.matrix(df)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
        fig = sns.heatmap(matrix_df, vmin=0, vmax=1, cmap="Blues", linewidths=2, annot=True, annot_kws={"fontsize":5})
        fig.set_xticklabels(['CD117', 'CD11B', 'CD13', 'CD19', 'CD33', 'CD34', 'CD38', 'CD45', 'CD56', 'CD7', 'FSC-A', 'FSC-H', 'HLA-DR', 'SSC-A', 'y'], fontsize=7)
        fig.set_yticklabels(['CD117', 'CD11B', 'CD13', 'CD19', 'CD33', 'CD34', 'CD38', 'CD45', 'CD56', 'CD7', 'FSC-A', 'FSC-H', 'HLA-DR', 'SSC-A', 'y'], fontsize=7)
        # fig.set_xticklabels(['CD117', 'CD11B', 'CD13', 'CD33', 'CD34', 'CD38', 'CD45', 'CD56', 'CD7', 'FSC-A', 'FSC-H', 'HLA-DR', 'SSC-A', 'y'], fontsize=7)
        # fig.set_yticklabels(['CD117', 'CD11B', 'CD13', 'CD33', 'CD34', 'CD38', 'CD45', 'CD56', 'CD7', 'FSC-A', 'FSC-H', 'HLA-DR', 'SSC-A', 'y'], fontsize=7)
        fig.set_ylabel(' ')
        fig.set_xlabel(' ')
        for tick in fig.get_xticklabels():
            tick.set_fontweight('bold')
        for tick in fig.get_yticklabels():
            tick.set_fontweight('bold')
        fig = fig.get_figure()
        fig.savefig('PPSResults/001.png', dpi=600)
        matrix_df.to_excel('PPSResults/001.xlsx', index=True)

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
    parser.add_argument('--dataset', default='Data/DataInPatientsUmap', type=str, choices=['Data/UsefulData','Data/UsefulData002', 'Data/DataInPatientsUmap'])
    parser.add_argument('-train', '--train', action='store_true')
    parser.add_argument('--test_model_path', default='Results/DNN-notShuffle-dropout0d5/DNN_Adam_98.23_checkpoint.t7', type=str)
    parser.add_argument('-shuffle', '--shuffle', action='store_true')
    args = parser.parse_args()

    object = AMLDataset(args)
    input, target, _ = object.__getitem__(0)
    print(input.shape)
    
    object.getPPScore()

    # ablation_dic_001 = {'SSC-A': [62.83185840707964, 0.8620053655264923], 'FSC-A': [80.53097345132744, 0.9057679409792085], 'FSC-H': [60.176991150442475, 0.9076123407109322], 'CD7': [62.83185840707964, 0.8253688799463448], 'CD11B': [78.76106194690266, 0.9618544600938967], 'CD13': [82.30088495575221, 0.9858316566063046], 'CD19': [81.85840707964601, 0.9936284372904092], 'CD33': [81.85840707964601, 0.9891851106639838], 'CD34': [42.0353982300885, 0.9897719651240777], 'CD38': [89.38053097345133, 0.9771965124077799], 'CD45': [83.1858407079646, 0.9746814218645203], 'CD56': [95.13274336283186, 0.9866700201207242], 'CD117': [71.68141592920354, 0.9539738430583502], 'HLA-DR': [91.15044247787611, 0.9315057008718981]}
    # ablation_dic_002 = {'SSC-A': [48.95104895104895, 0.9585127201565559], 'FSC-A': [65.03496503496504, 0.9906066536203523], 'FSC-H': [65.03496503496504, 0.9886497064579256], 'CD7': [79.02097902097903, 0.997651663405088], 'CD11B': [92.3076923076923, 0.9818003913894324], 'CD13': [90.9090909090909, 0.9927592954990214], 'CD33': [50.34965034965035, 0.9473581213307241], 'CD34': [62.93706293706294, 0.6796477495107632], 'CD38': [54.54545454545455, 0.3273972602739726], 'CD45': [65.03496503496504, 0.585518590998043], 'CD56': [100.0, 1.0], 'CD117': [91.60839160839161, 0.9972602739726028], 'HLA-DR': [86.01398601398601, 0.9937377690802348]}
    # key_list = []
    # value_list = []
    # for key in ablation_dic_002.keys():
    #     if key in ablation_dic_001.keys():
    #         key_list.append(key)
    #         score = 100-((ablation_dic_001[key][0]+ablation_dic_002[key][0]/2)/3*2+(100*ablation_dic_001[key][1]+100*ablation_dic_002[key][1]/2)/3*2)/2
    #         value_list.append(score)
    
    # sorted_id = sorted(range(len(value_list)), key=lambda k: value_list[k], reverse=True)
    # for item in sorted_id:
    #     print(key_list[item], value_list[item])