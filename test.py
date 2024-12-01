import pandas as pd
import numpy as np

if __name__ == '__main__':
    data = np.load('Data/npyData/proceededData.npz')
    print(data['X'].shape)