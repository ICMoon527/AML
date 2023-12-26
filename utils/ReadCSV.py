import pandas as pd
import datatable as dt
import numpy as np
from xpinyin import Pinyin
import os
import shutil
import logger

dict = {}

class ReadCSV():  # 2285 * 15
    def __init__(self, filepath='Data/FinalSheet.csv') -> None:
        self.data = dt.fread(filepath).to_pandas()
        self.P = Pinyin()
        self.saved_folder = 'Data/PickedCSV'
        self.useful_data_folder = 'Data/UsefulData'
        if not os.path.exists(self.saved_folder):
            os.mkdir(self.saved_folder)
        if not os.path.exists(self.useful_data_folder):
            os.mkdir(self.useful_data_folder)

        self.all = {'CD34 APC-A', 'CD79a APC-A', 'PerCP-Cy5-5-A', 'HL-DR V450-A', 'MPO PE-A', 'BV605-A', 'PE-Cy7-A', 'CD235 PE-A', 
                    'CD123 APC-Cy7-A', 'CD8 FITC-A', 'CD117 APC-Cy7-A', 'CD5 PerCP-Cy5-5-A', 'HLA-DR APC-A', 'CD79A APC-A', 'CD38 V450-A', 
                    'CD117 APC-A', 'CD64 APC-Cy7-A', 'CD99 PE-A', 'cCD3 APC-Cy7-A', 'CD11B BV605-A', 'CD10 APC-R700-A', 'CD38 APC-Cy7-A', 
                    'CD15 V450-A', 'CD19 FITC-A', 'CD64 FITC-A', 'cCD79a APC-A', 'CD56 FITC-A', 'CD19/CD56 FITC-A', 'CD71 FITC-A', 
                    'CD15 FITC-A', 'CD3 APC-Cy7-A', 'CD33 PE-Cy7-A', 'CD34 PE-A', 'CD11b BV605-A', 'V450-A', 'CD3 APC-A', 'CD19+CD56 FITC-A', 
                    'CD123 APC-R700-A', 'CD117 PerCP-Cy5-5-A', 'CD14 APC-Cy7-A', 'cCD3 APC-A', 'CD19/CD56/CD15 FITC-A', 'CD85D PE-Cy7-A', 
                    'CD20 APC-Cy7-A', 'CD15 BV605-A', '11b BV605-A', 'CD9 FITC-A', 'APC-R700-A', 'CD13 PE-A', 'APC-Cy7-A', 'MPO FITC-A', 
                    'CD16 V450-A', 'CD56 APC-R700-A', 'CD22 PE-A', 'CD321 FITC-A', 'CD13 PerCP-Cy5-5-A', 'CD2 APC-A', 'HLA-DR V450-A', 
                    'CD64 PE-A', 'DR V450-A', 'CD45 V500-C-A', 'CD71 APC-A', 'CD7 APC-R700-A', 'CD11B V450-A', 'cCD79A PE-A', 'CD8 APC-R700-A', 
                    'CD36 FITC-A', 'CD14 APC-A', 'CD16 APC-Cy7-A', 'CD56/CD19 FITC-A', 'CD4 V450-A'}
        
        self.intersection = {'CD7 APC-R700-A', 'CD11B BV605-A', 'CD19+CD56 FITC-A', 'CD117 PerCP-Cy5-5-A', 'DR V450-A', 'CD45 V500-C-A', 
                             'CD34 APC-A', 'CD38 APC-Cy7-A', 'CD13 PE-A', 'CD33 PE-Cy7-A'}

    def chooseNeed(self):  # pick M2 M4 and M5 up in the final sheet.
        nrows, ncols = self.data.shape
        for i in range(nrows):
            statement = self.data['临床诊断'][i]
            name = self.data['姓名'][i]
            if ('M2' in statement) or ('m2' in statement) or ('M5' in statement) or ('m5' in statement) or ('M4' in statement) or ('m4' in statement):
                if ('腰痛' not in statement) and ('M4/M5' not in statement):  # 去掉不明确的项
                    name_pinyin = self.P.get_pinyin(name).replace('-', '')  # 名字变拼音并去掉横杠
                    dict[name_pinyin] = statement
        self.countNum()
    
    def countNum(self):
        M2_count, M5_count = 0, 0
        for key in dict.keys():
            if 'M2' in dict[key]:
                M2_count += 1
            elif 'M5' in dict[key]:
                M5_count += 1
        print(M2_count, M5_count)

    def findSameProteinAndSaveFile(self, path):
        intersection = {'CD7 APC-R700-A', 'CD11B BV605-A', 'CD19+CD56 FITC-A', 'CD117 PerCP-Cy5-5-A', 'DR V450-A', 'CD45 V500-C-A', 
                        'CD34 APC-A', 'CD38 APC-Cy7-A', 'CD13 PE-A', 'CD33 PE-Cy7-A'}
        all = {'CD34 APC-A', 'CD79a APC-A', 'PerCP-Cy5-5-A', 'HL-DR V450-A', 'MPO PE-A', 'BV605-A', 'PE-Cy7-A', 'CD235 PE-A', 'CD123 APC-Cy7-A', 
               'CD8 FITC-A', 'CD117 APC-Cy7-A', 'CD5 PerCP-Cy5-5-A', 'HLA-DR APC-A', 'CD79A APC-A', 'CD38 V450-A', 'CD117 APC-A', 'CD64 APC-Cy7-A', 
               'CD99 PE-A', 'cCD3 APC-Cy7-A', 'CD11B BV605-A', 'CD10 APC-R700-A', 'CD38 APC-Cy7-A', 'CD15 V450-A', 'CD19 FITC-A', 'CD64 FITC-A', 
               'cCD79a APC-A', 'CD56 FITC-A', 'CD19/CD56 FITC-A', 'CD71 FITC-A', 'CD15 FITC-A', 'CD3 APC-Cy7-A', 'CD33 PE-Cy7-A', 'CD34 PE-A', 
               'CD11b BV605-A', 'V450-A', 'CD3 APC-A', 'CD19+CD56 FITC-A', 'CD123 APC-R700-A', 'CD117 PerCP-Cy5-5-A', 'CD14 APC-Cy7-A', 'cCD3 APC-A', 
               'CD19/CD56/CD15 FITC-A', 'CD85D PE-Cy7-A', 'CD20 APC-Cy7-A', 'CD15 BV605-A', '11b BV605-A', 'CD9 FITC-A', 'APC-R700-A', 'CD13 PE-A', 
               'APC-Cy7-A', 'MPO FITC-A', 'CD16 V450-A', 'CD56 APC-R700-A', 'CD22 PE-A', 'CD321 FITC-A', 'CD13 PerCP-Cy5-5-A', 'CD2 APC-A', 
               'HLA-DR V450-A', 'CD64 PE-A', 'DR V450-A', 'CD45 V500-C-A', 'CD71 APC-A', 'CD7 APC-R700-A', 'CD11B V450-A', 'cCD79A PE-A', 
               'CD8 APC-R700-A', 'CD36 FITC-A', 'CD14 APC-A', 'CD16 APC-Cy7-A', 'CD56/CD19 FITC-A', 'CD4 V450-A'}
        all_protein = set()
        
        if 'Extracted' in path:
            for root, dirs, files in os.walk(path):
                for file in files:
                    data = dt.fread(os.path.join(root, file)).to_pandas()
                    all = all | set(data.columns)
                for item in all:
                    if ' ' in item:
                        protein_name = item.split(' ')[0]
                        all_protein.add(protein_name)  # 加入集合
                    else:
                        # 没有空格说明这个通道没有放蛋白标记
                        continue
                print('All_protein: ', all_protein)  # 并集
                intersection_protein = all_protein.copy()  # 拷贝值
                for file in files:
                    file_protein = set()
                    data = dt.fread(os.path.join(root, file)).to_pandas()
                    for item in set(data.columns):
                        if ' ' in item:
                            protein_name = item.split(' ')[0]
                            file_protein.add(protein_name)  # 加入集合
                        else:
                            # 没有空格说明这个通道没有放蛋白标记
                            continue
                    intersection_protein = intersection_protein & file_protein
                print('Intersection_protein: ', intersection_protein)  # 交集

                    # 另存为相同蛋白荧光的流式文件
                    # if len(set(data.columns)&all) >= 10 and len(set(data.columns)&intersection) >= 8:
                    #     print(len(set(data.columns)&all), len(set(data.columns)&intersection), file, 'SAVED')
                    #     shutil.copy(os.path.join(root, file), os.path.join(self.saved_folder, file))
                    # else:
                    #     print(len(set(data.columns)&all), len(set(data.columns)&intersection), file, 'DISCARDED')
        
        elif 'Picked' in path:
            M2_num, M5_num, M2_10_num, M5_10_num = 0, 0, 0, 0
            for root, dirs, files in os.walk(path):
                for file in files:
                    data = dt.fread(os.path.join(root, file)).to_pandas()
                    if 'M2' in file:
                        M2_num += 1
                        if len(set(data.columns) & intersection) == 10:
                            if not os.path.exists(os.path.join(self.useful_data_folder, file)):
                                # Data/UsefulData
                                shutil.copy(os.path.join(root, file), os.path.join(self.useful_data_folder, file))
                            M2_10_num += 1
                    elif 'M5' in file:
                        M5_num += 1
                        if len(set(data.columns) & intersection) == 10:
                            if not os.path.exists(os.path.join(self.useful_data_folder, file)):
                                shutil.copy(os.path.join(root, file), os.path.join(self.useful_data_folder, file))
                            M5_10_num += 1
                    else:
                        print('ERROR!!!')
                        exit()
                    
                    # {'DR V450-A', 'CD19+CD56 FITC-A'} 通常都是少这俩
                    print(len(set(data.columns) & all), len(set(data.columns) & intersection), intersection-set(data.columns), file)
                # Total file nums: 88, M2 num: 17/36, M5 num: 20/52
                print('Total file nums: {}, M2 num: {}/{}, M5 num: {}/{}'.format(len(files), M2_10_num, M2_num, M5_10_num, M5_num))
                    
            # print(self.all&all)
    
    def readUseful(self, length=10000):
        length = int(length)
        useful_protein = ['CD19+CD56 FITC-A', 'CD13 PE-A','CD117 PerCP-Cy5-5-A', 'CD33 PE-Cy7-A', 'CD34 APC-A', 'CD7 APC-R700-A',
       'CD38 APC-Cy7-A', 'DR V450-A', 'CD45 V500-C-A', 'CD11B BV605-A']
        X, Y = list(), list()
        m2_logger = logger.setup_logger('M2_log', 'Doc/', 0, 'M2_log.txt')
        m5_logger = logger.setup_logger('M5_log', 'Doc/', 0, 'M5_log.txt')
        for root, dirs, files in os.walk(self.useful_data_folder):
            for file in files:
                data = dt.fread(os.path.join(root, file)).to_pandas()
                numpy_data = pd.DataFrame(data, columns=useful_protein).to_numpy()

                # if 'M2' in file:
                #     m2_logger.info(data.describe())
                # elif 'M5' in file:
                #     m5_logger.info(data.describe())
                # else:
                #     print('ERROR')
                #     exit()

                # 归一化
                numpy_data[numpy_data<0] = 0.
                numpy_data[numpy_data>1023] = 1023.
                numpy_data = numpy_data/1023.
                # 舍去长度小于 length 的数据
                if numpy_data.shape[0] < length:
                    continue
                else:
                    for i in range(int(numpy_data.shape[0]/10000.)):
                        slice = numpy_data[i*length:(i+1)*length, :]
                        X.append(slice)
                        if 'M2' in file:
                            Y.append(0)
                        elif 'M5' in file:
                            Y.append(1)
                        else:
                            print('ERROR')
                            exit()

                

        return np.array(X), np.array(Y)

object = ReadCSV('Data/FinalSheet.csv')
object.chooseNeed()
# print(dict)

if __name__ == '__main__':
    # object.findSameProteinAndSaveFile('Data/PickedCSV')
    X, Y = object.readUseful()
    print(X.shape, Y.shape)