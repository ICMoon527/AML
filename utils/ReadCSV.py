import pandas as pd
import datatable as dt
import numpy as np
from xpinyin import Pinyin
import os
import shutil
if __name__ == '__main__':
    import logger
else:
    from utils import logger
import json

dic = {}
m2_logger = logger.setup_logger('M2_log', 'Doc/', 0, 'M2_log.txt')
m5_logger = logger.setup_logger('M5_log', 'Doc/', 0, 'M5_log.txt')

class ReadCSV():  # 2285 * 15
    def __init__(self, filepath='Data/FinalSheet.csv') -> None:
        if os.path.exists(filepath):
            self.data = dt.fread(filepath).to_pandas()
        self.P = Pinyin()
        self.saved_folder = 'Data/PickedCSV'
        self.useful_data_folder = 'Data/UsefulData'
        if not os.path.exists(self.saved_folder):
            os.mkdir(self.saved_folder)
        if not os.path.exists(self.useful_data_folder):
            os.mkdir(self.useful_data_folder)

        self.all = {'FSC-W', 'CD15 BV605-A', 'CD64 PE-A', 'FSC-A', 'DR V450-A', 'CD123 APC-R700-A', 'MPO PE-A', 'CD11B V450-A', 'SSC-W', 'CD45 V500-C-A', 
                    'CD71 FITC-A', 'HLA-DR APC-A', 'cCD3 APC-Cy7-A', 'CD56 FITC-A', 'CD123 APC-Cy7-A', 'CD19/CD56/CD15 FITC-A', 'FSC-H', 'CD19/CD56 FITC-A', 
                    'CD34 PE-A', 'HLA-DR V450-A', 'HL-DR V450-A', 'CD2 APC-A', 'CD36 FITC-A', 'PE-A', 'CD19 FITC-A', 'MPO FITC-A', 'BV605-A', 'FITC-A', 
                    'CD235 PE-A', 'CD5 PerCP-Cy5-5-A', 'CD117 PerCP-Cy5-5-A', 'CD14 APC-A', 'Time', 'CD14 APC-Cy7-A', 'cCD79A PE-A', 'cCD3 APC-A', 
                    'CD16 APC-Cy7-A', 'CD22 PE-A', 'CD38 APC-Cy7-A', 'APC-R700-A', 'CD64 APC-Cy7-A', 'CD71 APC-A', 'CD7 APC-R700-A', 'CD15 V450-A', 
                    'PerCP-Cy5-5-A', 'cCD79a APC-A', 'CD56/CD19 FITC-A', 'CD9 FITC-A', 'APC-Cy7-A', 'CD15 FITC-A', 'CD64 FITC-A', 'CD10 APC-R700-A', 'SSC-A', 
                    'PE-Cy7-A', 'CD34 APC-A', 'CD79A APC-A', 'CD3 APC-A', 'CD4 V450-A', 'CD3 APC-Cy7-A', 'V450-A', '11b BV605-A', 'CD20 APC-Cy7-A', 'CD19+CD56 FITC-A', 
                    'CD33 PE-Cy7-A', 'SSC-H', 'CD13 PE-A', 'CD8 FITC-A', 'CD11B BV605-A', 'CD79a APC-A', 'CD117 APC-A', 'CD13 PerCP-Cy5-5-A', 'CD56 APC-R700-A', 
                    'CD8 APC-R700-A', 'CD11b BV605-A', 'CD16 V450-A'}
        
        self.all_physics = set()
        
        self.intersection = set()
        
        self.all_protein = {'CD20', 'FSC-A', 'CD15', 'CD2', 'HL-DR', 'CD22', 'FSC-H', 'CD235', 'CD11b', 'CD19/CD56/CD15', 'CD3', 'CD123', 'CD7', 'HLA-DR', 
                            '11b', 'DR', 'CD38', 'CD13', 'MPO', 'cCD79A', 'CD10', 'CD56', 'cCD3', 'CD36', 'CD8', 'CD19/CD56', 'CD79A', 'FSC-W', 'CD11B', 
                            'CD19+CD56', 'SSC-W', 'CD56/CD19', 'SSC-H', 'CD45', 'CD117', 'CD9', 'CD19', 'CD4', 'cCD79a', 'CD16', 'CD33', 'SSC-A', 'CD79a', 
                            'CD14', 'CD64', 'CD71', 'CD34', 'CD5'}
        
        self.intersection = set()
        
        self.intersection_protein, self.intersection_protein_1, self.intersection_protein_2, self.intersection_protein_3, self.intersection_protein_4, self.intersection_protein_5 = \
        self.all_protein.copy(), self.all_protein.copy(), self.all_protein.copy(), self.all_protein.copy(), self.all_protein.copy(), self.all_protein.copy()

        self.merge_protein, self.merge_protein_1, self.merge_protein_2, self.merge_protein_3, self.merge_protein_4, self.merge_protein_5 = (
        set(), 
        {'CD45', 'CD19/CD56', 'CD19', 'CD34', 'CD33', 'CD56/CD19', 'CD16', 'CD15', 'CD19+CD56', 'CD9', 'CD38', 'CD14', 'CD13', 'DR', 'CD7', 'CD56', 'CD19/CD56/CD15', 'CD11B', 'CD123', 'HL-DR', 'HLA-DR', 'CD64', 'CD117'}, 
        {'CD45', 'CD34', 'CD33', 'CD56/CD19', 'CD15', 'CD16', 'CD19+CD56', 'CD9', 'CD38', 'CD14', 'CD36', 'CD13', 'DR', 'CD7', 'CD56', '11b', 'CD19/CD56/CD15', 'CD11B', 'CD123', 'HL-DR', 'HLA-DR', 'CD10', 'CD64', 'CD117'}, 
        {'CD45', 'CD19/CD56', 'CD34', 'CD33', 'CD56/CD19', 'CD22', 'CD15', 'CD19+CD56', 'CD4', 'CD5', 'CD38', 'CD13', 'CD8', 'DR', 'CD7', 'CD56', 'CD3', 'CD19/CD56/CD15', 'CD11B', 'CD20', 'HLA-DR', 'CD117'}, 
        {'HLA-DR', 'CD45', 'CD34', 'CD33', 'CD15', 'CD56/CD19', 'CD19+CD56', 'CD9', 'CD38', 'CD13', 'CD71', 'DR', 'CD7', 'CD11B', 'CD123', 'CD235', 'CD2', 'CD117'}, 
        {'CD45', 'CD34', 'CD33', 'CD15', 'CD56/CD19', 'CD19+CD56', 'cCD79A', 'CD38', 'CD79A', 'CD13', 'cCD79a', 'cCD3', 'DR', 'CD7', 'CD3', 'CD19/CD56/CD15', 'CD11B', 'MPO', 'HLA-DR', 'CD79a', 'CD117'}
        )

        self.useful_items = {'FSC-A', 'FSC-H', 'SSC-A', 'CD45', 'CD19', 'CD34', 'CD33', 'CD38', 'CD13', 'DR', 'CD7', 'CD56', 'CD11B', 'HLA-DR', 'CD117', 'HL-DR'}
        
        self.file_count_1, self.file_count_2, self.file_count_3, self.file_count_4, self.file_count_5 = 0,0,0,0,0
        self.M2_file_count_1, self.M2_file_count_2, self.M2_file_count_3, self.M2_file_count_4, self.M2_file_count_5 = 0,0,0,0,0

    def chooseNeed(self):  # pick M2 M4 and M5 up in the final sheet.
        nrows, ncols = self.data.shape
        for i in range(nrows):
            statement = self.data['临床诊断'][i]
            name = self.data['姓名'][i]
            if ('M2' in statement) or ('m2' in statement) or ('M5' in statement) or ('m5' in statement) or ('M4' in statement) or ('m4' in statement):
                if ('腰痛' not in statement) and ('M4/M5' not in statement):  # 去掉不明确的项
                    name_pinyin = self.P.get_pinyin(name).replace('-', '')  # 名字变拼音并去掉横杠
                    dic[name_pinyin] = statement
        self.countNum()
    
    def countNum(self):
        M2_count, M4_count, M5_count = 0, 0, 0
        for key in dic.keys():
            if 'M2' in dic[key]:
                M2_count += 1
            elif 'M5' in dic[key]:
                M5_count += 1
            elif 'M4' in dic[key]:
                M4_count += 1
        print('M2_count: {}, M4_count: {}, M5_count: {}'.format(M2_count, M4_count, M5_count))

    def findSameProteinAndSaveFile(self, path):
        if 'Extracted' in path:
            for root, dirs, files in os.walk(path):
                for file in files:
                    if ('M2' in file) or ('M5' in file):
                        data = dt.fread(os.path.join(root, file)).to_pandas()
                        self.all = self.all | set(data.columns)

                for item in self.all:
                    if ' ' in item:
                        protein_name = item.split(' ')[0]
                        self.all_protein.add(protein_name)  # 加入集合
                    elif ('FSC' in item) or ('SSC' in item):
                        self.all_protein.add(item)  # 加入集合
                    else:
                        # 没有空格说明这个通道没有放蛋白标记，且排除了物理参数
                        continue
                print('All_protein: ', self.all_protein)  # 并集
                self.intersection_protein = self.all_protein.copy()  # 拷贝值

                for file in files:
                    if ('M2' in file) or ('M5' in file):
                        file_protein = set()
                        data = dt.fread(os.path.join(root, file)).to_pandas()
                        for item in set(data.columns):
                            if ' ' in item:
                                protein_name = item.split(' ')[0]
                                file_protein.add(protein_name)  # 加入集合
                            elif ('FSC' in item) or ('SSC' in item):
                                file_protein.add(item)
                            else:
                                # 没有空格说明这个通道没有放蛋白标记
                                continue
                        
                        # 先重点关注第一管数据
                        if '001' in file:
                            self.file_count_1 += 1
                            if 'M2' in file:
                                self.M2_file_count_1 += 1

                            self.merge_protein_1 = self.merge_protein_1 | file_protein
                            self.intersection_protein_1 = self.intersection_protein_1 & file_protein
                            m2_logger.info('File: {} (type 001), has {} public proteins according to all the 001 files.\nThey are {}\n'.format(file, len(self.merge_protein_1 & file_protein), self.merge_protein_1 & file_protein))
                            if not os.path.exists(os.path.join(self.useful_data_folder, file)):
                                shutil.copy(os.path.join(root, file), os.path.join(self.useful_data_folder, file))
                        elif '002' in file:
                            self.file_count_2 += 1
                            if 'M2' in file:
                                self.M2_file_count_2 += 1

                            self.merge_protein_2 = self.merge_protein_2 | file_protein
                            self.intersection_protein_2 = self.intersection_protein_2 & file_protein
                        elif '003' in file:
                            self.file_count_3 += 1
                            if 'M2' in file:
                                self.M2_file_count_3 += 1

                            self.merge_protein_3 = self.merge_protein_3 | file_protein
                            self.intersection_protein_3 = self.intersection_protein_3 & file_protein
                        elif '004' in file:
                            self.file_count_4 += 1
                            if 'M2' in file:
                                self.M2_file_count_4 += 1

                            self.merge_protein_4 = self.merge_protein_4 | file_protein
                            self.intersection_protein_4 = self.intersection_protein_4 & file_protein
                        elif '005' in file:
                            self.file_count_5 += 1
                            if 'M2' in file:
                                self.M2_file_count_5 += 1

                            self.merge_protein_5 = self.merge_protein_5 | file_protein
                            self.intersection_protein_5 = self.intersection_protein_5 & file_protein
                        
                        self.intersection_protein = self.intersection_protein & file_protein
                        # print('{} 里面的交集蛋白是 {}'.format(file, self.all_protein & file_protein))
                
                print('Intersection_protein: ', self.intersection_protein)  # 交集
                print('文件001数量: {}({}/{}), 交并: {}, {}'.format(self.file_count_1, self.M2_file_count_1, self.file_count_1-self.M2_file_count_1, self.intersection_protein_1, self.merge_protein_1))
                print('文件002数量: {}({}/{}), 交并: {}, {}'.format(self.file_count_2, self.M2_file_count_2, self.file_count_2-self.M2_file_count_2, self.intersection_protein_2, self.merge_protein_2))
                print('文件003数量: {}({}/{}), 交并: {}, {}'.format(self.file_count_3, self.M2_file_count_3, self.file_count_3-self.M2_file_count_3, self.intersection_protein_3, self.merge_protein_3))
                print('文件004数量: {}({}/{}), 交并: {}, {}'.format(self.file_count_4, self.M2_file_count_4, self.file_count_4-self.M2_file_count_4, self.intersection_protein_4, self.merge_protein_4))
                print('文件005数量: {}({}/{}), 交并: {}, {}'.format(self.file_count_5, self.M2_file_count_5, self.file_count_5-self.M2_file_count_5, self.intersection_protein_5, self.merge_protein_5))
                
                # for file in files:
                #     # 另存为相同蛋白荧光的流式文件
                #     if len(set(data.columns)&all) >= 10 and len(set(data.columns)&intersection) >= 8:
                #         print(len(set(data.columns)&all), len(set(data.columns)&intersection), file, 'SAVED')
                #         shutil.copy(os.path.join(root, file), os.path.join(self.saved_folder, file))
                #     else:
                #         print(len(set(data.columns)&all), len(set(data.columns)&intersection), file, 'DISCARDED')
        
        elif 'Picked' in path:
            M2_num, M5_num, M2_10_num, M5_10_num = 0, 0, 0, 0
            for root, dirs, files in os.walk(path):
                for file in files:
                    data = dt.fread(os.path.join(root, file)).to_pandas()
                    
                    if 'M2' in file:
                        M2_num += 1
                        if len(set(data.columns) & self.intersection) == 10:
                            if not os.path.exists(os.path.join(self.useful_data_folder, file)):
                                # Data/UsefulData
                                shutil.copy(os.path.join(root, file), os.path.join(self.useful_data_folder, file))
                            M2_10_num += 1
                    elif 'M5' in file:
                        M5_num += 1
                        if len(set(data.columns) & self.intersection) == 10:
                            if not os.path.exists(os.path.join(self.useful_data_folder, file)):
                                shutil.copy(os.path.join(root, file), os.path.join(self.useful_data_folder, file))
                            M5_10_num += 1
                    elif 'M4' in file:
                        pass
                    else:
                        print('ERROR!!!')
                        exit()
                    
                    # {'DR V450-A', 'CD19+CD56 FITC-A'} 通常都是少这俩
                    print(len(set(data.columns) & all), len(set(data.columns) & self.intersection), self.intersection-set(data.columns), file)
                # Total file nums: 88, M2 num: 17/36, M5 num: 20/52
                print('Total file nums: {}, M2 num: {}/{}, M5 num: {}/{}'.format(len(files), M2_10_num, M2_num, M5_10_num, M5_num))
                    
            # print(self.all&all)
                
    def saveAsNpy(self, file_name, df, cols):
        file_name = file_name[:-3]+'npy'
        data_np = np.zeros((len(cols)-1, df.shape[0]))  # HLA-DR和HL-DR是同一个，所以-1
        if 'HL-DR' in df.columns:  # 处理编辑错误的情况
            data_np[-1] = df['HL-DR'].to_numpy()
        # 处理多种混用的情况
        elif 'CD19/CD56' in df.columns:
            data_np[6] = df['CD19/CD56'].to_numpy()
            data_np[11] = df['CD19/CD56'].to_numpy()
        elif 'CD56/CD19' in df.columns:
            data_np[6] = df['CD56/CD19'].to_numpy()
            data_np[11] = df['CD56/CD19'].to_numpy()
        elif 'CD19+CD56' in df.columns:
            data_np[6] = df['CD19+CD56'].to_numpy()
            data_np[11] = df['CD19+CD56'].to_numpy()
        elif 'CD19/CD56/CD15' in df.columns:
            data_np[6] = df['CD19/CD56/CD15'].to_numpy()
            data_np[11] = df['CD19/CD56/CD15'].to_numpy()

        for i, col in enumerate(cols[0:-1]):
            if col in df.columns:
                data_np[i] = df[col].to_numpy()
            else:
                continue
        np.save(os.path.join(self.useful_data_folder, file_name), data_np)
        return data_np
    
    def readUseful(self):
        useful_items = ['SSC-A', 'FSC-A', 'FSC-H', 'CD7', 'CD11B', 'CD13', 'CD19', 'CD33', 'CD34', 'CD38', 'CD45', 'CD56', 'CD117', 'DR', 'HLA-DR',      'HL-DR']  # 16
        
        for root, dirs, files in os.walk(self.useful_data_folder):
            for file in files:
                if not 'csv' in file:
                    continue
                data = dt.fread(os.path.join(root, file)).to_pandas()
                new_columns_dict = dict()

                # 修改列名
                for item in data.columns:
                    if ' ' in item:
                        protein_name = item.split(' ')[0]
                        new_columns_dict[item] = protein_name
                    elif ('FSC' in item) or ('SSC' in item):
                        new_columns_dict[item] = item
                    else:
                        # 没有空格说明这个通道没有放蛋白标记
                        continue
                data.rename(columns=new_columns_dict, inplace=True)

                if 'M2' in file:
                    m2_logger.info(data.describe())
                elif 'M5' in file:
                    m5_logger.info(data.describe())
                else:
                    print('ERROR')
                    exit()

                # 将某个病人的数据存成.npy数组文件
                self.saveAsNpy(file, data, useful_items)
        
        return 0


    def getDataset(self, length=10000):
        length = int(length)
        X, Y = list(), list()
        # useful_items = ['SSC-A', 'FSC-A', 'FSC-H', 'CD7', 'CD11B', 'CD13', 'CD19', 'CD33', 'CD34', 'CD38', 'CD45', 'CD56', 'CD117', 'DR', 'HLA-DR',      'HL-DR']  # 16

        for root, dirs, files in os.walk(self.useful_data_folder):
            for file in files:
                if 'npy' in file:
                    numpy_data = np.load(os.path.join(root, file))
                    
                    # 归一化
                    numpy_data[numpy_data<0] = 0.
                    numpy_data[numpy_data>1023] = 1023.
                    # numpy_data = numpy_data/1023.
                    # 舍去长度小于 length 的数据
                    if numpy_data.shape[1] < length:
                        continue
                    else:
                        for i in range(int(numpy_data.shape[1]/10000.)):
                            slice = numpy_data[:, i*length:(i+1)*length]
                            X.append(slice)
                            if 'M2' in file:
                                Y.append(0)
                            elif 'M5' in file:
                                Y.append(1)
                            else:
                                print('ERROR')
                                exit()               

        return np.array(X), np.array(Y)

# ==================================================================
object = ReadCSV('Data/FinalSheet.csv')
object.chooseNeed()
print('病人类别字典: ', dic)

if __name__ == '__main__':
    # object.findSameProteinAndSaveFile('Data/ExtractedCSV')
    # object.findSameProteinAndSaveFile('Data/PickedCSV')
    # object.readUseful()
    X, Y = object.getDataset()
    print(X.shape, Y.shape, np.count_nonzero(Y==0), X.max())