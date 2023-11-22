import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Setting a random seed for reproducibility
random.seed(42)


def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        print(f'Dataframe shape: {df.shape}')
        print(f'Dataframe columns: {df.columns.tolist()}')
        df.info()
        return df

    except Exception as e:
        print(f'Error Loading data: {e}')
    return None


def check_and_clean_data(df):
    # Checking for null values
    null_counts = df.isnull().sum()
    if np.any(null_counts):
        print("Null values found:")
        print(null_counts[null_counts > 0])
    else:
        print("No null values found.")

    # Checking for duplicates
    duplicates = df.duplicated().sum()
    if duplicates:
        print(f"Number of duplicate rows: {duplicates}")
    else:
        print("No duplicate rows found.")

    # Dropping the 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in df.columns:
        print("Dropping the 'Unnamed: 0' column.")
        df = df.drop('Unnamed: 0', axis=1)

    # Checking for magic numbers or other anomalies in the data
    for column in df.columns:
        top_values = df[column].value_counts().head()
        print(f"\nMost frequent values in column '{column}':\n{top_values}")

    return df


# Encoding the 'Class' column considering the number of unique values
# (2 : "Benign" or "Attack") in the target column before encoding.
def encode_class_column(df, target_column, expected_ordinality):
    if df[target_column].nunique() != expected_ordinality:
        print(f"Expected target_column '{target_column}' to have "
              f"{expected_ordinality} values, got {df[target_column].nunique()}.")
    df[target_column] = df[target_column].map({'Benign': 0,
                                               'Attack': 1})
    return df


def plot_feature_correlation(df, features):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[features].corr(),
                annot=True,
                cmap='coolwarm',
                fmt='.2f',
                linewidths=0.1)
    plt.show()


def check_class_balance(df, target_column):
    class_counts = df[target_column].value_counts()
    print(f"Class balance:\n{class_counts}")
    print(f"Class ratio: {class_counts[0] / class_counts[1]:.2f}:1"
          f" (Class 0:Class 1)")


# Creating a balanced dataset by under sampling and split it into train,
# validation, and test sets.
def create_balanced_split(df, target_column, size_per_class, test_size=0.2,
                          val_size=0.1):
    # Getting the indices of all the rows belonging to each class
    ids_benign = df.index[df[target_column] == 0].tolist()
    ids_attack = df.index[df[target_column] == 1].tolist()

    # Checking if either class has fewer samples than the desired
    # size_per_class
    if len(ids_benign) < size_per_class or len(ids_attack) < size_per_class:
        size_per_class = min(len(ids_benign), len(ids_attack))
        print(f"Insufficient data in one of the classes. 'size_per_class' "
              f"adjusted to {size_per_class}.")

    # Now shuffling these two lists
    random.shuffle(ids_benign)
    random.shuffle(ids_attack)

    # Selecting the first 'size_per_class' of each category for the training
    # set
    train_ids = ids_benign[:size_per_class] + ids_attack[:size_per_class]

    # Creating an intermediate training set which will only used for training
    intermediate_train_df = df.loc[train_ids]

    # Remaining data will be part of the test set
    remaining_df = df.drop(train_ids)

    # Splitting the intermediate training set into training and validation sets
    train_df, val_df = \
        train_test_split(intermediate_train_df, test_size=val_size,
                         stratify=intermediate_train_df[target_column])

    # Splitting the remaining data into test set, preserving the class
    # distribution
    test_df, _ = train_test_split(remaining_df, test_size=test_size,
                                  stratify=remaining_df[target_column])

    return train_df, val_df, test_df


def visualize_feature_scales(df, features):
    df[features].plot(kind='box', figsize=(10, 8), vert=False)
    plt.title("Feature Scales")
    plt.show()


# Data Loading: load_data function will load the data from a CSV file and
# print its basic structure.
data_path = 'data/cicddos2019_dataset.csv'
df = load_data(data_path)

# Checking and Cleaning the Data: check_and_clean_data will check for null
# values,duplicates, and remove the 'Unnamed: 0' column if it exists.
df = check_and_clean_data(df)

# Checking the target('Class') unique values
print(f'Unique values in the target column "Class" are {df.Class.unique()}')

# Label Encoding
encode_class_column(df, target_column='Class', expected_ordinality=2)
print(f'Unique values in the target column "Class" after encoding are'
      f' {df.Class.unique()}')

# Check and print the balance of classes in the target variable.
check_class_balance(df, 'Class')

""" It's a very imbalanced dataset with 77% of the target class being an
 "Attack". I need to simplify my problem as much as possible by training my 
 model on the balanced dataset extracted from the original one. I will also 
 do some feature engineering to be able to find and use only the most relavent
  features to my target class in my training. As I have 80 columns, I will 
  separate them in sections of a size 10 - 12 features. I will then use my 
  knowledge of network packets and also the information from the correlation 
  matrix to choose the most relevant features.

"""

columns_batch_1 = ['Protocol', 'Flow Duration', 'Total Fwd Packets',
                   'Total Backward Packets', 'Fwd Packets Length Total',
                   'Bwd Packets Length Total', 'Fwd Packet Length Max',
                   'Fwd Packet Length Min', 'Fwd Packet Length Mean',
                   'Fwd Packet Length Std', 'Class']

columns_batch_2 = ['Bwd Packet Length Max', 'Bwd Packet Length Min',
                   'Bwd Packet Length Mean', 'Bwd Packet Length Std',
                   'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
                   'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Class']

columns_batch_3 = ['Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
                   'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total',
                   'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
                   'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Class']

columns_batch_4 = ['Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
                   'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
                   'Packet Length Min', 'Packet Length Max',
                   'Packet Length Mean', 'Packet Length Std',
                   'Packet Length Variance', 'FIN Flag Count', 'Class']

columns_batch_5 = ['SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
                   'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count',
                   'ECE Flag Count', 'Down/Up Ratio', 'Avg Packet Size',
                   'Avg Fwd Segment Size', 'Class']

columns_batch_6 = ['Avg Bwd Segment Size', 'Fwd Avg Bytes/Bulk',
                   'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
                   'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk',
                   'Bwd Avg Bulk Rate', 'Subflow Fwd Packets',
                   'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Class']

columns_batch_7 = ['Subflow Bwd Bytes', 'Init Fwd Win Bytes',
                   'Init Bwd Win Bytes', 'Fwd Act Data Packets',
                   'Fwd Seg Size Min', 'Active Mean', 'Active Std',
                   'Active Max', 'Active Min', 'Idle Mean', 'Idle Std',
                   'Idle Max', 'Idle Min', 'Label', 'Class']

column_batches = [columns_batch_1, columns_batch_2, columns_batch_3,
                  columns_batch_4,
                  columns_batch_5, columns_batch_6, columns_batch_7]

for batch in column_batches:
    plot_feature_correlation(df, batch)

"""Combining my domain knowledge and considering the correlation matrix, I'm 
keeping the most relevant features (top 10) to my target class:



1.   Protocol
2.   Fwd Packet Length mean : Mean size of packet in forward direction
3.   Flow Bytes/s : Number of flow bytes per second
4.   Packet Length Min : Minimum length of a packet
5.   Packet Length Mean : Mean length of a packet
6.   Avg Fwd Segment Size : Average  Segment size observed in the forward 
direction
7.   Avg Packet Size
8.   Sub flow Fwd Bytes : The average number of bytes in a sub flow in the 
forward direction . Which is the count of packets in a particular sub flow 
traveling from source to destination
9.   Fwd Act Data Packets : Count of packets with at least 1 byte of TCP 
data payload in the forward direction
10.  Flow IAT Mean : IAT (Inter-Arrival Time) is the time difference between 
the arrival times of two consecutive packets or flows
"""

columns_to_keep = ['Protocol', 'Fwd Packet Length Mean', 'Flow IAT Mean',
                   'Flow Bytes/s', 'Packet Length Mean', 'Packet Length Min',
                   'Avg Fwd Segment Size', 'Avg Packet Size',
                   'Subflow Fwd Bytes', 'Fwd Act Data Packets', 'Class']

df = df[columns_to_keep]
print(f'* df columns are: {df.columns} \n * Shape of the df is: {df.shape}')

""" Before I check to see if I need to do scaling, I will need to create a 
balanced dataset by under sampling and split it into train, validation, and 
test sets. I will use the function "create_balanced_split" which will create
a balanced training set with 5000 instances of each class, then out of that
training set, 10% will be used for validation, and the rest of the data 
(after removing the training instances) will be used for the test set."""

train_df, val_df, test_df = create_balanced_split(df, target_column='Class',
                                                  size_per_class=5000,
                                                  test_size=0.2,
                                                  val_size=0.1)

"""I would like to see if I need to scale the values or not. I will use the 
box plots for this purpose."""

visualize_feature_scales(df, df.columns)

"""Data scaling is necessary due to the significant variation in the range of 
different features. I will apply the scaler on my training, validation and 
test sets and will then save the scaler so that I an use it later on for 
consistency. I'm excluding the target "Class" as it doesn't need to be scaled."""

feature_columns_to_scale = columns_to_keep[:-1]

scaler = StandardScaler()
scaled_trained_data = scaler.fit_transform(train_df[feature_columns_to_scale])
scaled_train_df = pd.DataFrame(scaled_trained_data, columns=feature_columns_to_scale)

scaled_validation_data = scaler.fit_transform(val_df[feature_columns_to_scale])
scaled_validation_df = pd.DataFrame(scaled_validation_data, columns=feature_columns_to_scale)

scaled_test_data = scaler.fit_transform(test_df[feature_columns_to_scale])
scaled_test_df = pd.DataFrame(scaled_test_data, columns=feature_columns_to_scale)

# Reattaching the target column to the scaled feature DataFrames
scaled_train_df['Class'] = train_df['Class'].values
scaled_validation_df['Class'] = val_df['Class'].values
scaled_test_df['Class'] = test_df['Class'].values

# Saving the scaler
scaler_filename = "scalers/scaler.save"
joblib.dump(scaler, scaler_filename)

"""Now I'll check and make sure my training data is scaled."""

visualize_feature_scales(scaled_train_df, scaled_train_df.columns)

visualize_feature_scales(scaled_validation_df, scaled_validation_df.columns)

"""Now I can save my training, validation and test data so that I can load
 them later on."""

file_path = 'data/train_df.csv'
scaled_train_df.to_csv(file_path, index=False)

file_path = 'data/validation_df.csv'
scaled_validation_df.to_csv(file_path, index=False)

file_path = 'data/test_df.csv'
scaled_test_df.to_csv(file_path, index=False)

"""Now I can proceed with training the models."""
