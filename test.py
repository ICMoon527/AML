import pandas as pd
import numpy as np

if __name__ == '__main__':
    dic = np.load('Results/Test/AUC_dic.npy', allow_pickle=True).item()
    for key in dic.keys():
        fpr, tpr = dic[key]
        print(key, fpr.shape, tpr.shape)