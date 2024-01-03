from sklearn.cluster import KMeans
import numpy as np
import pandas as pd


def perform_clustering(scaled_data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(scaled_data)

    data_with_cluster = pd.concat([scaled_data, pd.DataFrame({'CLUSTER': kmeans.labels_})], axis=1)
    data_with_cluster.head(10)

    return kmeans, data_with_cluster


def compute_inertia_and_inter_cluster_distances(kmeans_model):
    # Tính toán Inertia (intra-cluster distance)
    inertia = kmeans_model.inertia_
    print("Inertia (Intra-cluster distance):", inertia)

    # Tính toán Inter-cluster distances
    cluster_centers = kmeans_model.cluster_centers_
    inter_cluster_distances = []

    for i in range(len(cluster_centers)):
        for j in range(i + 1, len(cluster_centers)):
            distance = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
            inter_cluster_distances.append(distance)

    avg_inter_cluster_distance = np.mean(inter_cluster_distances)
    print("Average Inter-cluster distance:", avg_inter_cluster_distance)

    # Trả về các giá trị tính toán
    return inertia, avg_inter_cluster_distance
