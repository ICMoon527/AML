import pandas as pd
from utils.ReadCSV import ReadCSV
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def ReduceDimensionUMAP():

    # Step 1: Load your CSV file (your clean data)
    object = ReadCSV()
    X, Y = object.getDataset('Data/UsefulData', length=10000)
    # X = X / 1023.
    X = X.transpose(0, 2, 1)  # (num, 10000, 15)
    X, Y = X.reshape((-1, X.shape[-1])), np.repeat(Y, repeats=10000)   ###############################

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=np.random.seed(1234))  # 固定种子是个好习惯啊！
    print('训练集长度: {}, 测试集长度: {}'.format(len(X_train), len(X_test)))
    print(f"Data Shape: {X_train.shape}")

    X_train = pd.DataFrame(X_train, columns=['SSC-A', 'FSC-A', 'FSC-H', 'CD7', 'CD11B', 'CD13', 'CD19', 'CD33', 'CD34', 'CD38', 'CD45', 'CD56', 'CD117', 'DR', 'HLA-DR'])
    print(X_train.describe())

    # Step 2: Standardize the features

    X_train = X_train / 1023.
    print(f"Data Shape After Scaling: {X_train.shape}")
    print(X_train.describe())

    # Step 3: Apply UMAP
    min_dists = [0, 0.01, 0.05, 0.1, 0.5, 1]
    n_neighbors = [5, 15, 30, 50, 100]
    for min_dist in min_dists:
        for n_neighbor in n_neighbors:

            if min_dist == 0:
                if n_neighbor != 100:
                    continue

            print('Now proceeding: n_neighbor {}, min_dist {}'.format(n_neighbor, min_dist))

            umap_reducer = umap.UMAP(n_neighbors=n_neighbor, min_dist=min_dist, n_components=2, random_state=42)
            significant_data = umap_reducer.fit_transform(X_train)
            np.save('Data/npyData/UMAP_Data_{}_{}.npy'.format(n_neighbor, str(min_dist).replace('.', '')), significant_data)
            print('Data/npyData/UMAP_Data_{}_{}.npy SAVED'.format(n_neighbor, str(min_dist).replace('.', '')))

            # Step 4: Prepare UMAP plot with 1st and 2nd features
            umap_df = pd.DataFrame(significant_data, columns=['UMAP1', 'UMAP2'])
            umap_df['Label'] = Y_train  # Add the group labels (M2: 0, M5: 1)

            # # Step 5: Plot UMAP
            plt.figure(figsize=(8, 5))
            sns.scatterplot(x='UMAP1', y='UMAP2', hue='Label', data=umap_df, palette={0: 'blue', 1: 'green'}, s=50)
            plt.savefig('UnsupResults/Sajid/UMAP_{}_{}.png'.format(n_neighbor, str(min_dist).replace('.', '')))
            print('UnsupResults/Sajid/UMAP_{}_{}.png SAVED'.format(n_neighbor, str(min_dist).replace('.', '')))

def ReduceDimensionANOVA():
    from scipy.stats import f_oneway

    # Load your CSV file (your clean data)
    object = ReadCSV()
    X, Y = object.getDataset('Data/UsefulData', length=10000)
    X = X.transpose(0, 2, 1)  # (num, 10000, 15)
    X, Y = X.reshape((-1, X.shape[-1])), np.repeat(Y, repeats=10000)   ###############################

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=np.random.seed(1234))  # 固定种子是个好习惯啊！
    data = np.load('Data/npyData/UMAP_Data_100_0.npy')  # data(X_train) and Y_train

    # Specify the features and the group column
    features = ['UMAP1', 'UMAP2']  #these are sepsis features but you can replace with your features
    group_column = ['Label']  # Replace with the actual name of your group column

    # Perform ANOVA for each feature
    data = np.hstack((data, Y_train.reshape((-1, 1))))
    significant_features = []
    p_value_threshold = 0.05  # Significance level ( that is level, you can change with 0.01 if you want)
    df = pd.DataFrame(data, columns=['UMAP1', 'UMAP2', 'Label'])  # (9016000, 3)
    print(df.head())

    # print("ANOVA Results:")
    # for feature in df.columns[:-1]:  # 排除'Label'列
    #     group1 = df[df['Label'] == 0][feature]
    #     group2 = df[df['Label'] == 1][feature]
        
    #     # Perform ANOVA
    #     f_stat, p_val = f_oneway(group1, group2)
        
    #     print(f"{feature}: F-statistic = {f_stat:.4f}, P-value = {p_val:.4f}")
        
    #     # Check if the p-value is less than the significance level
    #     if p_val < p_value_threshold:
    #         significant_features.append(feature)

    # # Display significant features
    # print("\nSignificant Features (p < 0.05): {}".format(significant_features))

    # # reserve significant data
    # reduced_df = df[significant_features + ['Label']]
    # print(reduced_df.shape)

    sample_size = 3500000
    n_samples_per_group = sample_size // 2
    group_0_indices = np.where(df['Label'] == 0)[0]
    group_1_indices = np.where(df['Label'] == 1)[0]

    # 从每种病人中随机抽取样本, 还能平衡样本数量
    sampled_indices_0 = np.random.choice(group_0_indices, size=n_samples_per_group, replace=False)
    sampled_indices_1 = np.random.choice(group_1_indices, size=n_samples_per_group, replace=False)

    # 合并样本
    sampled_indices = np.concatenate([sampled_indices_0, sampled_indices_1])
    reduced_data = df.iloc[sampled_indices].drop(columns=['Label']).values
    reduced_patient_labels = df.iloc[sampled_indices]['Label'].values

    return reduced_data, reduced_patient_labels


if __name__ == '__main__':
    # ReduceDimensionUMAP()
    reduced_data, reduced_patient_labels = ReduceDimensionANOVA()
    print(reduced_data.shape, reduced_patient_labels.shape)