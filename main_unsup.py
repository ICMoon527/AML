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

from utils.ReadCSV import ReadCSV
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import os
import tqdm
from dask.distributed import Client, progress
import dask
import dask.array as da
from pycirclize import Circos

def parallel_silhouette_score(data, labels, batch_size=10000):
    n_samples = len(data)
    batches = [slice(i, min(i + batch_size, n_samples)) for i in range(0, n_samples, batch_size)]
    # 定义延迟任务
    tasks = [dask.delayed(silhouette_samples)(data[slice_obj], labels[slice_obj]) for slice_obj in batches]
    # 执行所有任务
    results = dask.compute(*tasks)
    # 将结果拼接成一个数组
    all_scores = np.concatenate(results)
    # 计算平均轮廓系数
    silhouette_avg = np.mean(all_scores)
    
    return silhouette_avg

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

def BaseClusteringModel(type, X_train, draw=False):
    if type in 'MiniBatchKMeans':
        distortions = []
        sses = []
        silhouette_scores = []
        calinski_harabasz_scores = []
        davies_bouldin_scores = []
        # K = range(2, 20)
        K = [16]
        for k in K:
            print('=================================== K = {} =========================================='.format(k))
            #分别构建各种K值下的聚类器
            Model = MiniBatchKMeans(n_clusters=k, n_init='auto', verbose=1, batch_size=100000)
            Model.fit(X_train)

            #计算各个样本到其所在簇类中心欧式距离(保存到各簇类中心的距离的最小值)
            distortions.append(sum(np.min(cdist(X_train, Model.cluster_centers_, 'euclidean'), axis=1)) / X_train.shape[0])
            
            # print(Model.labels_)
            # print(Model.cluster_centers_)

            sses.append(Model.inertia_)
            silhouette_scores.append(parallel_silhouette_score(X_train, Model.labels_))
            calinski_harabasz_scores.append(parallel_calinski_harabasz_score(X_train, Model.labels_))
            davies_bouldin_scores.append(parallel_davies_bouldin_score(X_train, Model.labels_))

            if len(K) == 1:
                if draw:
                    draw_K(Model.labels_, X_train, Model.cluster_centers_, K[0])
                    print('Drawing Finished')
                return Model.labels_

        plt.plot(K, sses, label='WCSS', marker='o', linestyle='-', linewidth=1)
        plt.xlabel('optimal K')
        plt.ylabel('WCSS')
        plt.savefig('UnsupResults/KMeans/2-20.png')
        plt.close()

        plt.plot(K, silhouette_scores, label='silhouette', marker='o', linestyle='-', linewidth=1)
        plt.xlabel('optimal K')
        plt.ylabel('silhouette_score')
        plt.savefig('UnsupResults/KMeans/2-20_silhouette_score.png')
        plt.close()

        plt.plot(K, calinski_harabasz_scores, label='calinski_harabasz', marker='o', linestyle='-', linewidth=1)
        plt.xlabel('optimal K')
        plt.ylabel('calinski_harabasz_score')
        plt.savefig('UnsupResults/KMeans/2-20_calinski_harabasz_score.png')
        plt.close()

        plt.plot(K, davies_bouldin_scores, label='davies_bouldin', marker='o', linestyle='-', linewidth=1)
        plt.xlabel('optimal K')
        plt.ylabel('davies_bouldin_score')
        plt.savefig('UnsupResults/KMeans/2-20_davies_bouldin_score.png')
        plt.close()
    

    elif type in 'SpectralClustering':
        # X_train = X_train[0:100]
        distortions = []
        sses = []
        K = range(5,11)
        for k in K:
            print('=================================== K = {} =========================================='.format(k))
            #分别构建各种K值下的聚类器
            Model = SpectralClustering(n_clusters=k, verbose=1, eigen_solver='amg')
            Model.fit(X_train)

            #计算各个样本到其所在簇类中心欧式距离(保存到各簇类中心的距离的最小值)
            distortions.append(sum(np.min(cdist(X_train, Model.cluster_centers_, 'euclidean'), axis=1)) / X_train.shape[0])
            
            print(Model.labels_)
            print(Model.cluster_centers_)

            inertia = Model.inertia_
            sses.append(inertia)

        plt.plot(K,distortions,'bx-')
        #设置坐标名称
        plt.xlabel('optimal K')
        plt.ylabel('SSE')
        plt.savefig('UnsupResults/SpectralClustering/5-11.png')

        print(sses)
        print(distortions)

    elif type in 'GMM':
        from sklearn.mixture import GaussianMixture
        distortions = []

        K = range(5,21)
        for k in K:
            Model = GaussianMixture(n_components=k, verbose=2).fit(X_train)
            distortions.append(sum(np.min(cdist(X_train, Model.means_, 'euclidean'), axis=1)) / X_train.shape[0])

            print(Model.means_)

        plt.plot(K,distortions,'bx-')
        #设置坐标名称
        plt.xlabel('optimal K')
        plt.ylabel('SSE')
        plt.savefig('UnsupResults/GaussianMixture/5-20.png')

        print(distortions)

    elif type in 'DBSCAN':
        distortions = []

        # K = range(5,21)
        Model = DBSCAN(eps = 0.3, min_samples = 2).fit(X_train)
        # distortions.append(sum(np.min(cdist(X_train, Model.means_, 'euclidean'), axis=1)) / X_train.shape[0])

        print(min(Model.labels_), max(Model.labels_))

        # plt.plot(K,distortions,'bx-')
        # #设置坐标名称
        # plt.xlabel('optimal K')
        # plt.ylabel('SSE')
        # plt.savefig('UnsupResults/GaussianMixture/5-20.png')

        # print(distortions)


def is_point_in_ellipse(x0, y0, h, k, a, b):
    return ((x0 - h)**2 / a**2 + (y0 - k)**2 / b**2) < 1

def discardPointsOutsideEllipse(reduced_data, restored_data, h, k, a, b, angle):
    dist_ratio = ((reduced_data[:, 0] - h)**2 / a**2) + ((reduced_data[:, 1] - k)**2 / b**2)
    in_ellipse = dist_ratio < 1
    reduced_data = reduced_data[in_ellipse]
    restored_data = restored_data[in_ellipse]

    from matplotlib.patches import Ellipse
    ellipse = Ellipse(xy=(h,k), width=2*a, height=2*b, angle=angle, edgecolor='red', facecolor='none')
    plt.gca().add_artist(ellipse)

    # 绘制数据点
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.5)
    plt.title('PCA De-noised Data with 95% Confidence Ellipse')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.axis('equal') # 确保椭圆看起来是正确的比例
    plt.savefig('UnsupResults/Ellipse_new.png')

    return restored_data

def PCADenoising(X, Y, n_components=10):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from sklearn.decomposition import PCA
    from scipy.stats import chi2
    from sklearn.preprocessing import StandardScaler

    data = X
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # 使用PCA进行降噪
    pca = PCA(n_components=n_components) # 选择保留至少95%的方差
    pca.fit(data)
    reduced_data = pca.transform(data)

    # 数据恢复至原维度
    restored_data = pca.inverse_transform(reduced_data)

    # # 绘制降噪后的数据
    # plt.figure(figsize=(10, 8))

    # 95%置信度椭圆计算（仅针对前两个主成分）
    mean = np.mean(reduced_data[:, :2], axis=0)
    cov = np.cov(reduced_data[:, :2], rowvar=False)
    chi2_val = chi2.ppf(0.95, 2) # 2自由度下95%置信水平的卡方分布值
    eig_vals, eig_vecs = np.linalg.eig(cov)
    angle = np.degrees(np.arctan2(eig_vecs[1,0], eig_vecs[0,0])) # 主成分角度
    width, height = 2 * np.sqrt(chi2_val * eig_vals) # 椭圆的宽度和高度

    # 绘制95%置信度椭圆
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor='red', facecolor='none')
    plt.gca().add_artist(ellipse)

    # 绘制数据点
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.5)
    plt.title('PCA De-noised Data with 95% Confidence Ellipse')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.axis('equal') # 确保椭圆看起来是正确的比例
    plt.savefig('UnsupResults/Ellipse.png')
    plt.clf()

    restored_data = discardPointsOutsideEllipse(reduced_data, restored_data, mean[0], mean[1], width/2., height/2., angle)
    return restored_data

def hierarchicalClustering(X, useMiniBatch=True, batch_size=10000):
    from scipy.cluster.hierarchy import dendrogram, linkage
    from sklearn.cluster import AgglomerativeClustering
    from sklearn import metrics

    # 使用scikit-learn进行层次聚类
    centers = []
    if useMiniBatch:
        for i in range(0, len(X), batch_size):
            print('正在拟合第{}个Batch'.format(i/batch_size+1))
            # 获取当前批次数据
            X_batch = X[i:i+batch_size]
            
            # 使用层次聚类对批次数据进行聚类
            cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=10)
            cluster.fit(X_batch)
            
            # 计算每个聚类的中心
            labels = cluster.labels_
            for label in np.unique(labels):
                center = np.mean(X_batch[labels == label], axis=0)
                centers.append((label, center))

        # 将聚类中心转换为数组
        centers_array = np.array([center for _, center in centers])
        X = centers_array

        # 在所有聚类中心上再次进行层次聚类
        final_clustering = AgglomerativeClustering(n_clusters=10)
        y_pred = final_clustering.fit_predict(X)

        # 输出最终的聚类结果
        labels = final_clustering.labels_
        # print("Final cluster labels:", final_labels)
    
    else:
        X = X[:100000, :]
        cluster = AgglomerativeClustering(linkage='ward', n_clusters=6)
        print('正在拟合')
        y_pred = cluster.fit_predict(X)
        labels = cluster.labels_

    # 绘制聚类树
    print('开始画树')
    plt.figure(figsize=(10, 7))
    Z = linkage(X, 'ward')  # 进行层次聚类
    dn = dendrogram(Z)
    plt.title('Hierarchical Clustering Dendrogram (truncated)')
    plt.savefig('UnsupResults/HierarchicalClustering/tree.png')
    plt.clf()

    # 绘制聚类后的数据点分布
    print('正在绘制分布')
    plt.scatter(X[:,0], X[:,1], c=y_pred, s=50, cmap='viridis')
    plt.title('Hierarchical Clustering Result')
    plt.savefig('UnsupResults/HierarchicalClustering/distribution.png')

    # 计算轮廓系数
    silhouette_score = metrics.silhouette_score(X, labels)  # 接近1最好

    # 计算Calinski-Harabasz Index
    calinski_harabasz_score = metrics.calinski_harabasz_score(X, labels)  # 越大越好

    # 计算Davies-Bouldin Index
    davies_bouldin_score = metrics.davies_bouldin_score(X, labels)  # 越小越好

    print(f'Silhouette Score: {silhouette_score}')
    print(f'Calinski Harabasz Score: {calinski_harabasz_score}')
    print(f'Davies Bouldin Score: {davies_bouldin_score}')

    return 0


def compute_co_occurrence(cluster_labels, n_clusters):
    # 检查输入是否有效
    if len(cluster_labels) == 0:
        raise ValueError("The input 'cluster_labels' cannot be empty.")

    # 初始化共现矩阵
    co_occurrence_matrix = np.zeros((n_clusters, n_clusters), dtype=int)

    # 计算共现矩阵
    for i in range(len(cluster_labels)):
        for j in range(i + 1, len(cluster_labels)):
            label_i = cluster_labels[i]
            label_j = cluster_labels[j]
            if label_i != label_j:  # 如果需要包含自身共现，则移除此行
                co_occurrence_matrix[label_i, label_j] += 1
                co_occurrence_matrix[label_j, label_i] += 1  # 确保矩阵是对称的

    return co_occurrence_matrix


if __name__ == '__main__':
    client = Client()  # 创建Dask本地客户端

    # Load your CSV file (your clean data)
    object = ReadCSV()
    X, Y = object.getDataset('Data/UsefulData', length=10000)
    X = X.transpose(0, 2, 1)  # (num, 10000, 15)
    X, Y = X.reshape((-1, X.shape[-1])), np.repeat(Y, repeats=10000)   ###############################

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False, random_state=np.random.seed(1234))  # 固定种子是个好习惯啊！
    X_train = np.load('Data/npyData/UMAP_Data_100_0.npy')  # data(X_train) and Y_train

    # Specify the features and the group column
    features = ['UMAP1', 'UMAP2']
    group_column = ['Label']
    
    cluster_labels = BaseClusteringModel('MiniBatchKMeans', X_train, draw=False)
    n_clusters = max(cluster_labels)+1

    # Step 3: Create Co-occurrence Matrix using Dask

    # 确保所有标签都在有效的范围内
    unique_labels = np.unique(cluster_labels)

    # 计算共现矩阵
    co_occurrence_matrix = compute_co_occurrence(cluster_labels, n_clusters)
    print(co_occurrence_matrix)

    print('Step 3 finished...')

    # Step 4: Convert Co-occurrence Matrix to DataFrame for Visualization
    row_names = [f"Cluster {i}" for i in range(1, n_clusters + 1)]
    co_occurrence_df = pd.DataFrame(co_occurrence_matrix, index=row_names, columns=row_names)

    # Step 5: Create and Display Chord Diagram using pycirclize
    circos = Circos.initialize_from_matrix(
        co_occurrence_df,
        space=2,
        r_lim=(93, 100),
        cmap="tab10",
        ticks_interval=100000,
        label_kws=dict(r=94, size=12, color="white"),
    )

    # Plot the Chord Diagram
    fig = circos.plotfig()
    plt.savefig('UnsupResults/KMeans/ChordDiagram.png')
    plt.close()

    client.close()