import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess_data(data):

    # Drop unnecessary columns
    drops = ["ADDRESSLINE2", "STATE", "TERRITORY", "PHONE", "ADDRESSLINE1", "POSTALCODE", "ORDERDATE",
             'STATUS', 'CITY', 'CONTACTFIRSTNAME', 'CONTACTLASTNAME', 'CUSTOMERNAME', 'ORDERNUMBER']
    data = data.drop(drops, axis=1)

    # Normalize data
    labelencoder = LabelEncoder()
    data.loc[:, 'PRODUCTLINE'] = labelencoder.fit_transform(data.loc[:, 'PRODUCTLINE'])
    data['COUNTRY'] = labelencoder.fit_transform(data['COUNTRY'])
    data['DEALSIZE'] = labelencoder.fit_transform(data['DEALSIZE'])
    data['PRODUCTCODE'] = labelencoder.fit_transform(data['PRODUCTCODE'])

    # Standardize data
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    return scaled_data

def select_highly_correlated_columns(data, threshold=0.9):
    correlation_matrix = data.corr()

    highly_correlated_columns = set()

    # Loop through the correlation matrix and select columns with high correlation values
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                colname = correlation_matrix.columns[i]
                highly_correlated_columns.add(colname)

    # Remove columns with high correlation
    df_selected = data.drop(columns=highly_correlated_columns)

    return df_selected

