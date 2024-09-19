import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv('UnsupResults/HierarchicalClustering/denoised_data.csv')
    print(data.describe())