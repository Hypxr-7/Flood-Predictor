import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch

def load_flood_data(file_path="data/final/combined-data.csv", test_size=0.2):
    df = pd.read_csv(file_path)

    # Encode categorical variables
    df['Month'] = df['Month'].astype('category').cat.codes
    df['Region'] = LabelEncoder().fit_transform(df['Region'])
    df['Flood'] = df['Flood'].map({'No': 0, 'Yes': 1})

    # Features and target
    X = df[['Year', 'Month', 'Region', 'Avg LST', 'Avg NDSI', 'Avg NDVI', 'Avg Precipitation']].values
    y = df['Flood'].values

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
