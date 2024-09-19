import numpy as np
import pandas as pd

# Load the data
data = np.load("/data1/lizehai/denoised_data.npy")



import pandas as dd
# import dask.dataframe as dd
# df = pd.DataFrame(data)

data = dd.read_csv(r"dinoised_data.csv", encoding = 'ISO-8859-1', blocksize=32e6)


data.isnull().sum().compute()
