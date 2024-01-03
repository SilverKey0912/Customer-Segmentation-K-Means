import pandas as pd

from data_processing import *
from visualization import *
from clustering import *
import joblib


def main():
    data = pd.read_csv(r'C:\Users\lymin\Documents\Semester 7\Data Mining\Final_project\Data\sales_data_sample.csv')

    # Thống kê mô tả về dữ liệu
    data.head()
    describe = data.describe()
    print(describe)
    data.info()
    check_null_data = check_null(data)
    print(check_null_data)

    # Tiền xử lý dữ liệu
    processed_data = load_and_preprocess_data(data)
    df_select = select_highly_correlated_columns(processed_data)

    # Hiển thị biểu đồ
    plot_pairplot(df_select)
    plot_pairplot(df_select)
    plot_bar_chart(df_select, 'COUNTRY', 'COUNTRY', 'Biểu Đồ Tương quan đơn đặt hàng giữa các quốc gia', 'Quốc Gia', 'Số Lượng')
    plot_distribution(df_select, 'PRICEEACH', 'Biểu đồ phân phối giá của mỗi sản phẩm', 'Price Ordered', 'Frequency')
    plot_distribution(df_select, 'SALES', 'Biểu đồ phân phối doanh thu của đơn hàng', 'Sales', 'Frequency')
    plot_correlation_heatmap(df_select)

    # Thực hiện phân cụm
    kmeans, data_with_cluster = perform_clustering(df_select, 5)

    # Trực quan hóa dữ liệu sau khi phân cụm và đánh giá độ chính xác
    visualize_pca_3d(kmeans, data_with_cluster)
    inertia, avg_inter_cluster_distance = compute_inertia_and_inter_cluster_distances(kmeans)
    silhouette_avg = silhouette_score(data_with_cluster, kmeans.labels_)
    print(f"Silhouette Score: {silhouette_avg}")
    print("Average Inter-cluster distance of 5 cluster:", avg_inter_cluster_distance)
    print("Inertia (Intra-cluster distance of 5 cluster):", inertia)

    # Lưu mô hình
    joblib.dump(kmeans, 'kmeans_model.joblib')


if __name__ == "__main__":
    main()
