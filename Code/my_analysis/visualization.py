import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np


def check_null(data):
    # Kiểm tra giá trị null
    nan_check = pd.isnull(data).sum()

    # Sắp xếp các biến null theo giảm dần
    null_variables = nan_check.sort_values(ascending=False)

    # Tính tỷ lệ giá trị null
    null_rates = (nan_check / data.isnull().count()).sort_values(ascending=False)

    # Tạo DataFrame chứa số lượng và tỷ lệ giá trị null
    missing_data = pd.concat([null_variables, null_rates], axis=1, keys=['N of null', 'Rates'])

    return missing_data

def plot_pairplot(data):
    plt.figure(figsize=(12, 8))
    sns.pairplot(data)
    plt.suptitle('Biểu đồ phân tán giữa các cột', y=1.02)
    plt.show()


def plot_bar_chart(data, x_column, y_column, title, x_label, y_label):
    chart_title = title
    x_axis_label = x_label
    y_axis_label = y_label

    fig = plt.Figure(figsize=(12, 6))
    fig = px.bar(x=data[x_column].value_counts().index, y=data[x_column].value_counts(),
                 color=data[y_column].value_counts().index, height=600)

    fig.update_layout(title_text=chart_title, title_x=0.5)
    fig.update_layout(xaxis_title=x_axis_label, yaxis_title=y_axis_label)
    fig.show()


def plot_distribution(data, column, title, x_label, y_label):
    plt.figure(figsize=(9, 6))
    sns.distplot(data[column])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def plot_correlation_heatmap(data):
    correlation_matrix = data.corr()
    plt.figure(figsize=(10, 5))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Ma trận tương quan')
    plt.show()


def plot_silhouette(scaled_data):
    scores = []
    for num_clusters in range(2, 10):
        clusterer = KMeans(n_clusters=num_clusters)
        pred = clusterer.fit_predict(scaled_data)
        # centers = clusterer.cluster_centers_
        scores.append(silhouette_score(scaled_data, pred, metric='euclidean'))

    plt.plot(scores, 'b*-')
    plt.xticks(np.arange(len(scores)), np.arange(1, len(scores) + 1))
    plt.title('Silhouette Score')
    plt.xlabel('n of clusters')
    plt.ylabel('Scores')
    plt.show()

def visualize_pca_3d(kmeans, data_with_cluster):
    pca = PCA(n_components=3)
    principal_comp = pca.fit_transform(data_with_cluster)
    pca_df = pd.DataFrame(data=principal_comp, columns=['pca_1', 'pca_2', 'pca_3'])
    pca_df = pd.concat([pca_df, pd.DataFrame({'cluster': kmeans.labels_})], axis=1)

    # Showing
    fig = px.scatter_3d(pca_df, x='pca_1', y='pca_2', z='pca_3',
                        color='cluster', symbol='cluster', size_max=20, opacity=0.6)
    fig.show()
