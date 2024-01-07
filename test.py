import os
import pandas as pd

if __name__ == '__main__':
    df = pd.DataFrame({'a':[1,2,3],'b':[1,2,3]})
    print(df['a'].to_numpy())