import dask.array
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import cdist
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler

from utils.ReadCSV import ReadCSV
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import os
import tqdm
from dask.distributed import Client, progress
import dask
from dask import delayed
import dask.array as da
from pycirclize import Circos
import seaborn as sns
import joblib

def parallel_silhouette_score(data, labels, batch_size=10000):
    n_samples = len(data)
    batches = [slice(i, min(i + batch_size, n_samples)) for i in range(0, n_samples, batch_size)]
    
    # 定义延迟任务（使用 @delayed 装饰器）
    @delayed
    def process_batch(slice_obj):
        return silhouette_samples(data[slice_obj], labels[slice_obj])
    
    # 并行计算所有分块
    tasks = [process_batch(slice_obj) for slice_obj in batches]
    results = dask.compute(*tasks)  # 触发并行执行
    
    # 合并结果并计算平均值
    all_scores = np.concatenate(results)
    return np.mean(all_scores)


def parallel_calinski_harabasz_score(dask_data, labels):
    dask_data = dask.array.from_array(dask_data, chunks=(100000, -1))
    result = calinski_harabasz_score(dask_data.compute(), labels)
    return result

def parallel_davies_bouldin_score(dask_data, labels):
    dask_data = dask.array.from_array(dask_data, chunks=(100000, -1))
    result = davies_bouldin_score(dask_data.compute(), labels)
    return result

def draw_K(labels, data, centers, K):
    plt.figure(figsize=(10, 6))

    # 使用不同的颜色和标记来表示不同的簇
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap('tab20')  # 'tab20' 是一个适合分类数据的 colormap
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    for i, label in enumerate(unique_labels):
        # 获取属于当前簇的所有数据点
        cluster_data = data[labels == label]
        
        # 绘制这些数据点
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], 
                    label=f'Cluster {label}', alpha=0.7, c=colors[i], s=30)

    # 可选：绘制簇中心（仅适用于某些聚类算法，如 KMeans)
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=150, alpha=0.75, marker='X', label='Centers')

    # 添加标题和标签
    plt.title('Clustering Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    # 显示图例
    plt.legend()

    # 保存图形
    plt.savefig('UnsupResults/KMeans/ClusteringResults_K{}.png'.format(K))
    plt.close()

def KMeans(X_train, K, draw=False):
    distortions = []
    sses = []
    silhouette_scores = []
    calinski_harabasz_scores = []
    davies_bouldin_scores = []
    
    for k in K:
        print('=================================== K = {} =========================================='.format(k))
        #分别构建各种K值下的聚类器
        Model = MiniBatchKMeans(n_clusters=k, n_init='auto', verbose=1, batch_size=100000)
        Model.fit(X_train)
        # 保存模型
        # joblib.dump(Model, 'UnsupResults/KMeansFor{}/kmeans_model.skops'.format(k))

        if len(K) == 1:
            if draw:
                draw_K(Model.labels_, X_train, Model.cluster_centers_, K[0])
                print('Drawing Finished')
            return Model.labels_

        #计算各个样本到其所在簇类中心欧式距离(保存到各簇类中心的距离的最小值)
        # distortions.append(sum(np.min(cdist(X_train, Model.cluster_centers_, 'euclidean'), axis=1)) / X_train.shape[0])
        # sses.append(Model.inertia_)
        silhouette_scores.append(silhouette_score(X_train, Model.labels_))
        # calinski_harabasz_scores.append(parallel_calinski_harabasz_score(X_train, Model.labels_))
        # davies_bouldin_scores.append(parallel_davies_bouldin_score(X_train, Model.labels_))

        print('silhouette score list: ', silhouette_scores)

    # plt.plot(K, sses, label='WCSS', marker='o', linestyle='-', linewidth=1)
    # plt.xlabel('optimal K')
    # plt.ylabel('WCSS')
    # plt.savefig('UnsupResults/KMeans/2-20.png')
    # plt.close()

    plt.plot(K, silhouette_scores, label='silhouette', marker='o', linestyle='-', linewidth=1)
    plt.xlabel('optimal K')
    plt.ylabel('silhouette_score')
    plt.savefig('UnsupResults/KMeans/2-30_silhouette_score.png')
    plt.close()

    # plt.plot(K, calinski_harabasz_scores, label='calinski_harabasz', marker='o', linestyle='-', linewidth=1)
    # plt.xlabel('optimal K')
    # plt.ylabel('calinski_harabasz_score')
    # plt.savefig('UnsupResults/KMeans/2-20_calinski_harabasz_score.png')
    # plt.close()

    # plt.plot(K, davies_bouldin_scores, label='davies_bouldin', marker='o', linestyle='-', linewidth=1)
    # plt.xlabel('optimal K')
    # plt.ylabel('davies_bouldin_score')
    # plt.savefig('UnsupResults/KMeans/2-20_davies_bouldin_score.png')
    # plt.close()


def CountClusterInPatient(data_scaled, cluster_labels, patient_cell_num):
    # Evaluate clustering using silhouette score (higher is better)
    # silhouette_avg = silhouette_score(data_scaled, cluster_labels)
    # print(f"Silhouette Score: {silhouette_avg:.3f}")

    # Analyze clustering: Count number of cells per cluster for each patient
    n_clusters = max(cluster_labels)+1
    n_patients = len(patient_cell_num)
    cluster_counts = np.zeros((n_patients, n_clusters), dtype=int)

    start_idx = 0
    for patient_idx, n_cells_per_patient in patient_cell_num:
        end_idx = start_idx + n_cells_per_patient
        patient_cluster_labels = cluster_labels[start_idx:end_idx]
        start_idx = end_idx
        
        # Count number of cells in each cluster for this patient
        unique, counts = np.unique(patient_cluster_labels, return_counts=True)
        cluster_counts[patient_idx-1, unique] = counts

    # Convert cluster counts to a DataFrame for easier visualization
    cluster_counts_df = pd.DataFrame(cluster_counts, columns=[f"Cluster {i+1}" for i in range(n_clusters)],
                                    index=[f"Patient {i+1}" for i in range(n_patients)])

    # Display the cluster counts for each patient
    print(cluster_counts_df)

    # Optionally save the cluster labels for further analysis
    np.save("UnsupResults/KMeansFor{}/cluster_labels.npy".format(n_clusters), cluster_labels)
    cluster_counts_df.to_csv("UnsupResults/KMeansFor{}/cluster_counts.csv".format(n_clusters))
    plotHistogram(n_patients, cluster_counts, cluster_counts_df, n_clusters)

    return 0


def loadPatientScaledData():
    X_train = list()
    patient_cell_num = list()
    for root, dirs, files in os.walk('Data/DataInPatientsUmap'):
        for file in files:
            if 'npy' in file:
                print('Proceeding {}...'.format(file))
                numpy_data = np.load(os.path.join(root, file))
                X_train.append(numpy_data)
                patient_cell_num.append([int(file.split('_')[1]), numpy_data.shape[0]])

    X_train = np.vstack((X_train))
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(X_train)
    return data_scaled, patient_cell_num

def getPatientScaledDataXY(max_length=100000):
    X_train = list()
    patient_cell_num = list()
    Y_train = list()
    Umap_1_max = -10000
    Umap_1_min = 10000
    Umap_2_max = -10000
    Umap_2_min = 10000
    for root, dirs, files in os.walk('Data/DataInPatientsUmap'):
        for file in files:
            if 'npy' in file:
                print('Proceeding {}...'.format(file))
                numpy_data = np.load(os.path.join(root, file))  # shape均值267000

                Umap_1_max = Umap_1_max if Umap_1_max > np.max(numpy_data[:,0]) else np.max(numpy_data[:,0])
                Umap_1_min = Umap_1_min if Umap_1_min < np.min(numpy_data[:,0]) else np.min(numpy_data[:,0])
                Umap_2_max = Umap_2_max if Umap_2_max > np.max(numpy_data[:,1]) else np.max(numpy_data[:,1])
                Umap_2_min = Umap_2_min if Umap_2_min < np.min(numpy_data[:,1]) else np.min(numpy_data[:,1])

                patient_cell_num.append([int(file.split('_')[1]), numpy_data.shape[0]])
                # 截长补短
                while numpy_data.shape[0] >= max_length:
                    X_train.append(numpy_data[:max_length])
                    Y_train.append(int(file.split('_')[-1][0]))  # 0:M2, 1:M5
                    numpy_data = numpy_data[max_length:]
                if len(numpy_data) > 0:
                    X_train.append(numpy_data)
                    Y_train.append(int(file.split('_')[-1][0]))  # 0:M2, 1:M5
                
                # Y_train.append(int(file.split('_')[-1][0]))  # 0:M2, 1:M5

    # standarize
    for i in range(len(X_train)):
        X_train[i][:,0] = (X_train[i][:,0]-Umap_1_min)/(Umap_1_max-Umap_1_min)
        X_train[i][:,1] = (X_train[i][:,1]-Umap_2_min)/(Umap_2_max-Umap_2_min)
        X_train[i] = torch.tensor(X_train[i])

    X_train = torch.nn.utils.rnn.pad_sequence(X_train, batch_first=True, padding_value=0)
    return np.array(X_train), np.array(Y_train)

def plotHistogram(n_patients, cluster_counts, cluster_counts_df, n_clusters=3):
    patient_id = np.random.randint(0, n_patients)  # Random patient
    plt.bar(range(n_clusters), cluster_counts[patient_id])
    plt.xlabel('Cluster')
    plt.ylabel('Number of Cells')
    plt.title(f'Cluster Distribution for Patient {patient_id + 1}')
    plt.savefig('UnsupResults/KMeansFor{}/ClusterDistributionForPatient.png'.format(n_clusters), dpi=600)

    circos = Circos.initialize_from_matrix(
        cluster_counts_df,
        space=2,
        r_lim=(93, 100),
        cmap="tab20",
        # ticks_interval=100000,
        label_kws=dict(r=105, size=9, orientation='vertical', weight='bold', color="black"))
    fig = circos.plotfig()
    plt.savefig('UnsupResults/KMeansFor{}/ClusterDistributionForPatientCircos.png'.format(n_clusters), dpi=600)
    plt.close()


def CountClusterInCells(data_scaled, cluster_labels):
    data = pd.DataFrame(data_scaled, columns=['UMAP1', 'UMAP2'])
    data['Cluster'] = cluster_labels
    cluster_proportions = data['Cluster'].value_counts(normalize=True).sort_index()

    # Expected proportions (example values, adjust as needed)
    expected_proportions = [0.60, 0.20, 0.20]  # Example proportions for the 3 clusters

    # Calculate deviations from expected proportions
    deviation_data = []
    for cluster, proportion in enumerate(cluster_proportions):
        deviation = abs(proportion - expected_proportions[cluster])
        deviation_data.append({'Cluster': cluster, 'Deviation': deviation})

    # Convert to DataFrame
    deviation_df = pd.DataFrame(deviation_data)

    # Step 5: Create a Manhattan Plot
    plt.figure(figsize=(10, 6))

    # Define colors for clusters
    colors = sns.color_palette("husl", n_clusters)

    # sns.scatterplot(
    #     x='Cluster', 
    #     y='Deviation', 
    #     hue='Cluster', 
    #     palette=colors, 
    #     data=deviation_df, 
    #     legend=None, 
    #     s=100
    # )

    # # Add axis labels and title
    # plt.xlabel('Cluster')
    # plt.ylabel('Deviation from Expected Proportion')
    # plt.title('Manhattan Plot of Cluster Deviations')

    # # Highlight the expected proportions (optional)
    # plt.axhline(y=0, color='grey', linestyle='--')

    # plt.grid(True)
    from matplotlib.collections import PathCollection
    dm = pd.DataFrame(data_scaled)
    ax = sns.violinplot(x= 'Cluster', y = 'Deviation', data = deviation_df)
    for artist in ax.lines:
        artist.set_zorder(10)
    for artist in ax.findobj(PathCollection):
        artist.set_zorder(11)

    sns.stripplot(x="Cluster", y="Deviation", data= deviation_df, jitter=True, ax=ax)

    plt.savefig('UnsupResults/KMeansFor3/ClusterDeviationInCells.png', dpi=600)



if __name__ == '__main__':
    client = Client()  # 创建Dask本地客户端

    # Load your CSV file (your clean data)
    data_scaled, patient_cell_num = loadPatientScaledData()

    # Specify the features and the group column
    features = ['UMAP1', 'UMAP2']
    group_column = ['Label']
    
    
    cluster_labels = KMeans(data_scaled[::100], range(2, 30), draw=False)  # 聚类
    exit()
    n_clusters = max(cluster_labels)+1

    CountClusterInPatient(data_scaled, cluster_labels, patient_cell_num)
    # CountClusterInCells(data_scaled, cluster_labels)  # only use in K=3