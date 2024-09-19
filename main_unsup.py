import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering, DBSCAN
from scipy.spatial.distance import cdist
from sklearn.datasets import load_iris

from utils.ReadCSV import ReadCSV
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import os
import tqdm


def BaseClusteringModel(type, X_train):
    if type in 'MiniBatchKMeans':
        distortions = []
        sses = []
        K = range(5,101)
        for k in K:
            print('=================================== K = {} =========================================='.format(k))
            #分别构建各种K值下的聚类器
            Model = MiniBatchKMeans(n_clusters=k, n_init='auto', verbose=1, batch_size=100000)
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
        plt.savefig('UnsupResults/KMeans/5-100.png')

        print(sses)
        print(distortions)

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


if __name__ == '__main__':
    object = ReadCSV()
    X, Y = object.getDataset('Data/UsefulData', length=10000)
    # X = X / 1023.
    X = X.transpose(0, 2, 1)  # (num, 10000, 15)
    X, Y = X.reshape((-1, X.shape[-1])), np.repeat(Y, repeats=10000)   ###############################

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=np.random.seed(1234))
    print('训练集长度: {}, 测试集长度: {}'.format(len(X_train), len(X_test)))

    # BaseClusteringModel('DBSCAN', X_train)
    if not os.path.exists('UnsupResults/HierarchicalClustering/denoised_data.npy'):
        denoised_data = PCADenoising(X_train, Y_train)
        np.save('UnsupResults/HierarchicalClustering/denoised_data.npy', denoised_data)
    else:
        denoised_data = np.load('UnsupResults/HierarchicalClustering/denoised_data.npy')
        csv_data = pd.DataFrame(denoised_data, columns=['SSC-A', 'FSC-A', 'FSC-H', 'CD7', 'CD11B', 'CD13', 'CD19', 'CD33', 'CD34', 'CD38', 'CD45', 'CD56', 'CD117', 'DR', 'HLA-DR'])
        csv_data.to_csv('UnsupResults/HierarchicalClustering/denoised_data.csv', index=False)
        exit()
    
    # Clustering
    hierarchicalClustering(denoised_data)