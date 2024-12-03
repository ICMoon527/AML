import numpy as np
import pandas as pd
import dask.dataframe as dd
import dask.array as da
# import dask_ml.cluster
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, fcluster
from pycirclize import Circos
import matplotlib.pyplot as plt
from ReduceDataDimension import ReduceDimensionANOVA

def checkData():
    # data = dd.read_csv(r"UnsupResults/HierarchicalClustering/denoised_data.csv", encoding = 'ISO-8859-1', blocksize=32e6)
    # print(data.isnull().sum().compute())
    data = np.load('Data/npyData/UMAP_Data.npy')
    return data

if __name__ == '__main__':

    # Step 1: Dimensionality Reduction
    reduced_data, reduced_patient_labels = ReduceDimensionANOVA()  # (350000, 2)
    print('Read Data Successfully')
    chunks = (1000000, 15)

    # data_np = data.compute()
    # tsne = TSNE(n_components=2, random_state=42)
    # data_reduced = tsne.fit_transform(data_np)
    print('Step 1 finished.')

    # Step 2: Apply Hierarchical Clustering
    # Perform hierarchical clustering on the reduced data
    n_clusters = 3  # Define the number of clusters you want
    linkage_matrix = linkage(reduced_data, method='ward')  # (350000, 2)
    cluster_labels = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')
    print('Step 2 finished.')

    # Step 3: Create Co-occurrence Matrix using Dask
    cluster_labels_da = da.from_array(cluster_labels, chunks=(1000000,))
    co_occurrence_matrix = da.zeros((n_clusters, n_clusters), dtype=int)

    # Function to compute co-occurrence for a chunk
    def compute_co_occurrence(chunk, co_occurrence_matrix):
        for i in range(len(chunk)):
            for j in range(i + 1, len(chunk)):  # Avoid double counting pairs
                if chunk[i] != chunk[j]:
                    co_occurrence_matrix[chunk[i] - 1, chunk[j] - 1] += 1
                    co_occurrence_matrix[chunk[j] - 1, chunk[i] - 1] += 1
        return co_occurrence_matrix

    # Parallel computation of co-occurrence matrix using Dask
    co_occurrence_matrix = da.map_blocks(
        compute_co_occurrence,
        cluster_labels_da,
        co_occurrence_matrix,
        dtype=int,
)

    co_occurrence_matrix = co_occurrence_matrix.compute()  # Convert to NumPy after computation
    print('Step 3 finished.')

    # Step 4: Convert Co-occurrence Matrix to DataFrame for Visualization
    row_names = [f"Cluster {i}" for i in range(1, n_clusters + 1)]
    co_occurrence_df = pd.DataFrame(co_occurrence_matrix, index=row_names, columns=row_names)
    print('Step 4 finished.')

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
    plt.savefig('UnsupResults/Sajid/Chord_Diagram.png')
    print('Step 5 finished.')