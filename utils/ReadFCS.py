from FlowCytometryTools import FCMeasurement
import os
from ReadCSV import dict
import shutil

class FCSReader():
    def __init__(self) -> None:
        self.main_folder = 'Data/FCS'
        self.save_folder = 'Data/ExtractedFCS'
        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)


    def checkAllAndSaveNeeded(self):
        for root, dirs, files in os.walk(self.main_folder):
            for i, file in enumerate(files):
                if 'fcs' in file:
                    file_name = file.replace(' ', '').lower()  # 把名字中间的空格去掉并小写
                    for key in dict.keys():  # 匹配需要的病人姓名，遍历字典中的名字
                        if key in file_name:
                            if 'aml' in file_name:  # 只要AML的病人
                                sick_name = dict[key]
                                if ('M2' in sick_name) or ('m2' in sick_name):
                                    sick_name = 'M2_'
                                elif ('M5' in sick_name) or ('m5' in sick_name):
                                    sick_name = 'M5_'
                                elif ('M4' in sick_name) or ('m4' in sick_name):
                                    sick_name = 'M4_'
                                else:
                                    print('Error sick_name')
                                    exit()

                                path = os.path.join(root, file)
                                if os.path.exists(self.save_folder+'/'+sick_name+file):
                                    print('Existed!.')
                                    continue
                                # 把M2,M4,M5的病人文件另存为
                                shutil.copy(path, self.save_folder+'/'+sick_name+file)
                                print(path)
                        else:
                            continue
                else:
                    continue
            

    def fcs2csv(self):
        save_path = 'Data/ExtractedCSV'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for root, dirs, files in os.walk('Data/ExtractedFCS'):
            for file in files:
                if not '.fcs' in file:
                    continue
                fcs_file = FCMeasurement(ID='read', datafile=os.path.join(root, file))
                data = fcs_file.data
                print('PROCEEDING ', file)
                data.to_csv(os.path.join(save_path, file.split('.')[0]+'.csv'), index=False)


if __name__ == '__main__':
    reader = FCSReader()
    reader.checkAllAndSaveNeeded()
    reader.fcs2csv()