import glob
import os
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

def load_data(data_csv):
    """Load the combined dataset from CSV file"""
    if os.path.exists(data_csv):
        print(f"Loading existing {data_csv}")
        df = pd.read_csv(data_csv)
    else:
        df_list = []
        data_dir = 'data'
        csv_files = glob.glob(os.path.join(data_dir, '**', '*.csv'), recursive=True)
        print(f"Creating {data_csv }")
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
        df.to_csv(data_csv , index=False)
        
        # filter duplicate rows
        df_dedup = df.drop_duplicates()
        print(f"Merged data shape of deduped: {df_dedup.shape}")
        df_dedup.to_csv(data_csv.replace('.csv', '_dedup.csv'), index=False)
        print(f"Saved deduped data to {data_csv.replace('.csv', '_dedup.csv')}")
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
    combined = f"{source_ip}.{destination_ip}{source_port}{destination_port}{protocol}"
    return float(combined)


def prepro_data(config, df):
    """ Preprocess the data """
    # parameters based on FlowGuard paper
    train_data_size = config['split']['train_data_size']
    test_data_size = config['split']['test_data_size']

    df = df.sample(n = train_data_size + test_data_size, random_state=42).reset_index(drop=True)
    # df = df.sample(frac=0.00001, random_state=42).reset_index(drop=True)

    # Remove 'Flow ID' which is unique for each row
    # df = df.drop(['Flow ID'], axis=1)
    df['Flow ID'] = df['Flow ID'].apply(convert_to_float)


    # df_train = df.iloc[:train_data_size]
    # df_test = df.iloc[train_data_size:]

    # Encode categorical features
    label_encoder = LabelEncoder()
    df['Source IP'] = label_encoder.fit_transform(df['Source IP'])
    df['Destination IP'] = label_encoder.fit_transform(df['Destination IP'])


    # Split the data into features and target
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