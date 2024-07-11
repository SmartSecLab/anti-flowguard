import glob
import os
import requests
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from joblib import dump


# Features used in FlowGuard
selected_features = [
    "Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port", "Protocol",
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets",
    "Total Length of Bwd Packets", "Fwd Packet Length Max", "Fwd Packet Length Min",
    "Fwd Packet Length Mean", "Fwd Packet Length Std", "Bwd Packet Length Max",
    "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s", "Packet Length Std", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Active Mean", "Active Std",
    "Idle Mean", "Idle Std", "Label"
]


def download_zip_file(url, save_path):
    # Create the data directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Get the filename from the URL
    filename = url.split("/")[-1]

    # Full path to save the downloaded file
    file_path = os.path.join(save_path, filename)

    # Download the file
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Write the content to a file
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded and saved to {file_path}")
        return file_path
    else:
        print(f"Failed to download file. Status code: {response.status_code}")
        return None


def extract_zip_file(zip_file_path, extract_to_base):
    # Get the folder name by stripping the .zip extension from the file name
    folder_name = os.path.basename(zip_file_path).replace('.zip', '')
    extract_to = os.path.join(extract_to_base, folder_name)

    # Create the extraction directory if it doesn't exist
    os.makedirs(extract_to, exist_ok=True)

    # Extract the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"File extracted to {extract_to}")


def load_data(data_csv):
    """Load the combined dataset from CSV file"""
    if os.path.exists(data_csv):
        print(f"Loading existing {data_csv}")
        df = pd.read_csv(data_csv)
    else:
        df_list = []
        data_dir = 'data'
        # download_zip_file if CSV-01-12.zip does not exist
        if not os.path.exists(os.path.join(data_dir, 'CSV-01-12.zip')):
            print("Downloading CSV-01-12.zip...")
            first_file = 'http://205.174.165.80/CICDataset/CICDDoS2019/Dataset/CSVs/CSV-01-12.zip'
            zip_file_path = download_zip_file(first_file, data_dir)
            if zip_file_path:
                print("Extracting CSV-01-12.zip...")
                extract_zip_file(zip_file_path, data_dir)

        if not os.path.exists(os.path.join(data_dir, 'CSV-13-14.zip')):
            print("Downloading CSV-03-11.zip...")
            second_file = 'http://205.174.165.80/CICDataset/CICDDoS2019/Dataset/CSVs/CSV-03-11.zip'
            zip_file_path = download_zip_file(second_file, data_dir)
            if zip_file_path:
                print("Extracting CSV-03-11.zip...")
                extract_zip_file(zip_file_path, data_dir)
                print("Downloaded and extracted all files.")

        csv_files = glob.glob(os.path.join(
            data_dir, '**', '*.csv'), recursive=True)
        print(f"Creating {data_csv}")
        for data_file in csv_files:
            print("Loading dataset:", data_file)
            df_file = pd.read_csv(data_file, engine='pyarrow')
            df_file.columns = df_file.columns.str.strip()

            # pick only selected features
            df_file = df_file[selected_features]

            # Encode labels as binary (benign or not)
            # df_file['Label'] = df_file['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
            df_list.append(df_file)

        df = pd.concat(df_list, ignore_index=True)
        print(f"Merged data shape: {df.shape}")
        df.to_csv(data_csv, index=False)

        # filter duplicate rows
        df_dedup = df.drop_duplicates()
        print(f"Merged data shape of deduped: {df_dedup.shape}")
        dedup_path = data_csv.replace('.csv', '_dedup.csv')
        df_dedup.to_csv(dedup_path, index=False)
        print(f"Saved deduped data to {dedup_path}")
    return df


def ip_to_number(ip):
    octets = list(map(int, ip.split('.')))
    return (octets[0] * (256**3)) + (octets[1] * (256**2)) + (octets[2] * 256) + octets[3]


def convert_to_float(data):
    parts = data.split('-')
    source_ip = ip_to_number(parts[0])
    destination_ip = ip_to_number(parts[1])
    source_port = parts[2]
    destination_port = parts[3]
    protocol = parts[4]
    combined = "{source_ip}.{destination_ip}{source_port}{destination_port}{protocol}"
    return float(combined)


def prepro_data(config, df):
    """ Preprocess the data """
    # Remove 'Flow ID' which is unique for each row
    # df = df.drop(['Flow ID'], axis=1)
    df['Flow ID'] = df['Flow ID'].apply(convert_to_float)

    # df_train = df.iloc[:train_data_size]
    # df_test = df.iloc[train_data_size:]

    # Encode categorical features
    label_encoder = LabelEncoder()
    df['Source IP'] = label_encoder.fit_transform(df['Source IP'])
    df['Destination IP'] = label_encoder.fit_transform(df['Destination IP'])

    # Save the intermediate data
    df.to_csv(config['processed_data'], index=False)
    print(f"Saved preprocessed data to {config['processed_data']}")
    return df


def load_processed_data(config):
    """ Load the preprocessed data """
    data_csv = config['data']
    # check if config['processed_data'] exists
    if os.path.exists(config['processed_data']):
        data_csv = config['processed_data']
        print(f"Loading existing processed data: {data_csv}...")
        df = pd.read_csv(data_csv)
    else:
        print(f"Preprocessing data from {config['data']}...")
        df = load_data(data_csv)
        df = prepro_data(config, df)
        # parameters based on FlowGuard paper
    train_data_size = config['split']['train_data_size']
    test_data_size = config['split']['test_data_size']

    df = df.sample(n=train_data_size + test_data_size,
                   random_state=42).reset_index(drop=True)
    # df = df.sample(frac=0.00001, random_state=42).reset_index(drop=True)
    return df


def split_X_y(config, df):
    """ Split the data into train and test sets """
    # Encode labels as binary (benign or not)
    if config['class_type'] == 'binary':
        df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

    X = df.drop(columns=['Label'])
    X = X.astype(float)
    y = df['Label']

    # Replace infinite or very large values with NaN
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Impute NaN values with the mean of each feature
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=X.columns)
    print('-'*50)
    return X, y
