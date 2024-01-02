from flask import Flask, render_template, request
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import numpy as np

app = Flask(__name__)

# Đường dẫn đến file model.joblib
model_path = "C:/Users/lymin/Documents/Semester 7/Data Mining/Final_project/Code/kmeans_model.joblib"
kmeans_model = joblib.load(model_path)


# Route cho trang chủ
@app.route('/')
def index():
    return render_template('index.html')

# Route xử lý dự đoán khi nhận file CSV từ form
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        print("Predict route is called.")
        # Lấy file từ form
        uploaded_file = request.files['file']
        
        if uploaded_file.filename != '':
            # Đọc file CSV
            data = pd.read_csv(uploaded_file)

            data = data.drop(["ADDRESSLINE2","STATE","TERRITORY"], axis=1)
            data= data.drop(["PHONE","ADDRESSLINE1","POSTALCODE"],axis=1)
            data= data.drop(["ORDERDATE"],axis=1)
            drops = ['STATUS', 'CITY','CONTACTFIRSTNAME','CONTACTLASTNAME', 'CUSTOMERNAME', 'ORDERNUMBER']

            new_data= data.drop(drops,axis=1)

            labelencoder = LabelEncoder()
            new_data.loc[:, 'PRODUCTLINE'] = labelencoder.fit_transform(new_data.loc[:, 'PRODUCTLINE'])
            new_data['COUNTRY'] = labelencoder.fit_transform(new_data['COUNTRY'])
            new_data['DEALSIZE'] = labelencoder.fit_transform(new_data['DEALSIZE'])
            new_data['PRODUCTCODE'] = labelencoder.fit_transform(new_data['PRODUCTCODE'])

            scaler = StandardScaler()
            new_data_scaled = scaler.fit_transform(new_data)


            # Dự đoán cụm sử dụng mô hình đã được huấn luyện
            predicted_clusters = kmeans_model.predict(new_data_scaled)

            # Tạo DataFrame mới với dữ liệu đã được chuẩn hóa
            data_selected = pd.DataFrame(new_data_scaled, columns=new_data.columns)

            # Thêm cột 'CLUSTER' vào DataFrame mới
            data_with_clusters = pd.concat([data[['ORDERNUMBER']], pd.DataFrame({'CLUSTER': predicted_clusters})], axis=1)

            # Chuyển đổi kết quả thành HTML để hiển thị
            result_data = data_with_clusters[['ORDERNUMBER', 'CLUSTER']].to_dict(orient='records')

            return render_template('index.html', result_data=result_data)


    return 'Invalid file or no file provided.'

if __name__ == '__main__':
    app.run(debug=True)
