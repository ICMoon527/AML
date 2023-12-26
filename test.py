"""
数据量不够的情况下，通过杂糅打乱不同病人的切片流式再组合扩大数据量，与杂糅相同病人扩增数据集作对比，同一病人截不同长度的连续片段
"""
import datatable as dt
import shutil
from FlowCytometryTools import FCMeasurement
import os
import time

if __name__ == '__main__':
    for root, dirs, files in os.walk('Data\FCS'):
        for file in files:
            if '.fcs' in file:
                fcs_file = FCMeasurement(ID='read', datafile=os.path.join(root, file))
                data = fcs_file.data
                data.to_csv('test.csv', index=False)
                data = dt.fread('test.csv').to_pandas()
                print(data.max())
                print(os.path.join(root, file))