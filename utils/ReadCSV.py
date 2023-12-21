import pandas as pd
import datatable as dt
import numpy as np
from xpinyin import Pinyin
import os
import shutil

dict = {}

class ReadCSV():  # 2285 * 15
    def __init__(self, filepath) -> None:
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

    def chooseNeed(self):  # pick M2 and M5 up
        nrows, ncols = self.data.shape
        for i in range(nrows):
            statement = self.data['临床诊断'][i]
            name = self.data['姓名'][i]
            if ('M2' in statement) or ('m2' in statement) or ('M5' in statement) or ('m5' in statement):
                if '腰痛' not in statement:
                    name_pinyin = self.P.get_pinyin(name).replace('-', '')
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
        intersection = {'CD7 APC-R700-A', 'CD11B BV605-A', 'CD19+CD56 FITC-A', 'CD117 PerCP-Cy5-5-A', 'DR V450-A', 'CD45 V500-C-A', 'CD34 APC-A', 'CD38 APC-Cy7-A', 'CD13 PE-A', 'CD33 PE-Cy7-A'}
        all = {'CD34 APC-A', 'CD79a APC-A', 'PerCP-Cy5-5-A', 'HL-DR V450-A', 'MPO PE-A', 'BV605-A', 'PE-Cy7-A', 'CD235 PE-A', 'CD123 APC-Cy7-A', 'CD8 FITC-A', 'CD117 APC-Cy7-A', 'CD5 PerCP-Cy5-5-A', 'HLA-DR APC-A', 'CD79A APC-A', 'CD38 V450-A', 'CD117 APC-A', 'CD64 APC-Cy7-A', 'CD99 PE-A', 'cCD3 APC-Cy7-A', 'CD11B BV605-A', 'CD10 APC-R700-A', 'CD38 APC-Cy7-A', 'CD15 V450-A', 'CD19 FITC-A', 'CD64 FITC-A', 'cCD79a APC-A', 'CD56 FITC-A', 'CD19/CD56 FITC-A', 'CD71 FITC-A', 'CD15 FITC-A', 'CD3 APC-Cy7-A', 'CD33 PE-Cy7-A', 'CD34 PE-A', 'CD11b BV605-A', 'V450-A', 'CD3 APC-A', 'CD19+CD56 FITC-A', 'CD123 APC-R700-A', 'CD117 PerCP-Cy5-5-A', 'CD14 APC-Cy7-A', 'cCD3 APC-A', 'CD19/CD56/CD15 FITC-A', 'CD85D PE-Cy7-A', 'CD20 APC-Cy7-A', 'CD15 BV605-A', '11b BV605-A', 'CD9 FITC-A', 'APC-R700-A', 'CD13 PE-A', 'APC-Cy7-A', 'MPO FITC-A', 'CD16 V450-A', 'CD56 APC-R700-A', 'CD22 PE-A', 'CD321 FITC-A', 'CD13 PerCP-Cy5-5-A', 'CD2 APC-A', 'HLA-DR V450-A', 'CD64 PE-A', 'DR V450-A', 'CD45 V500-C-A', 'CD71 APC-A', 'CD7 APC-R700-A', 'CD11B V450-A', 'cCD79A PE-A', 'CD8 APC-R700-A', 'CD36 FITC-A', 'CD14 APC-A', 'CD16 APC-Cy7-A', 'CD56/CD19 FITC-A', 'CD4 V450-A'}
        
        if 'Extracted' in path:
            for root, dirs, files in os.walk(path):
                for file in files:
                    data = dt.fread(os.path.join(root, file)).to_pandas()
                    # all = all | set(data.columns)
                    # print(len(set(data.columns) & all), set(data.columns) & all, file)
                    print(len(set(data.columns)&all), len(set(data.columns)&intersection), file)

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

object = ReadCSV('Data/FinalSheet.csv')
object.chooseNeed()
# print(dict)

if __name__ == '__main__':
    object.findSameProteinAndSaveFile('Data/PickedCSV')