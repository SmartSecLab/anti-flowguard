import glob
import os
import yaml 
import pandas as pd
import numpy as np
from pathlib import Path


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

from loader import load_data, prepro_data, split_X_y


def load_config():
    """ Load the configuration from the YAML file """
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config


config = load_config()
print('-'*50)
print('Config:', config)
print('-'*50)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    # y_pred_class = classify(y_pred)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred, zero_division=0))
    # save classification report
    with open('classification_report.txt', 'w') as f:
        f.write(classification_report(y_test, y_pred, zero_division=0))
    print('-'*50)


def train_evaluate_models(X, y):
    """Train and evaluate classifiers"""
    # Split the data into training and testing sets
    # test_size = config['split']['test_size']
    test_size = config['split']['test_data_size'] / config['split']['train_data_size']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    classifiers = {
        "ID3 Classifier": DecisionTreeClassifier(),
        "Naive Bayes Classifier": GaussianNB(),
        "Random Forest Classifier": RandomForestClassifier(),
        # "Linear Regression": LinearRegression(),
    }

    for name, classifier in classifiers.items():
        print("Training", name, "...")
        classifier.fit(X_train, y_train)
        # Evaluate Linear Regression model for classification
        evaluate_model(classifier, X_test, y_test)
        
        # Save the model
        # dump(classifier, f'{name}_combined_model.joblib')
        # print("Model saved.")
    print('-'*50)

def plot_feature_importances(X, y):
    # Create a random forest classifier
    rf = RandomForestClassifier()

    # Fit the model to the data
    rf.fit(X, y)

    # Get feature importances
    importances = rf.feature_importances_

    # Get feature names
    feature_names = X.columns

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    sorted_feature_names = [feature_names[i] for i in indices]


    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), sorted_feature_names, rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
    plt.show()

    # save the figure 
    Path("figure").mkdir(parents=True, exist_ok=True)
    plt.savefig('figure/feature_importances.png')

    return sorted_feature_names


def load_processed_data():
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
    return df



# Load the data
df = load_processed_data()

# Split into X and y
X, y = split_X_y(config, df)

print('Training and evaluating with all features...')
# Train and evaluate models
train_evaluate_models(X, y)


# Plot feature importances
sorted_feature_names = plot_feature_importances(X, y)


# Train and evaluate only with the importent features
print('Training and evaluating with important features...')
train_evaluate_models(X, y)
