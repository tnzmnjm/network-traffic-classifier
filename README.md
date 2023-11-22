# Network Traffic Classifier
This project implements a network traffic classification system designed to identify different types of TCP/UDP based network attacks like NTP, DNS, LDAP, SNMP, and TFTP.


<img src="banner_photo.png" width="700" height="350">


## Description
The system includes Exploratory Data Analysis (EDA), pre-processing pipelines, Logistic Regression and XGBoost classification models, and evaluation metrics to assess performance.


## Dataset
The dataset used in this project is the CICDDoS2019, available on [Kaggle](https://www.kaggle.com/datasets/dhoogla/cicddos2019). The CICDDoS2019 dataset is created and maintained by the Canadian Institute for Cybersecurity (CIC). For more detailed feature descriptions, visit the [CIC's official dataset page](https://www.unb.ca/cic/datasets/ddos-2019.html).


## Methodology

The project utilizes the CICDDoS2019 dataset, which includes both benign and the most up-to-date common DDoS attacks, reflecting real-world network traffic data. The dataset encompasses labeled flows based on timestamps, source/destination IPs, ports, protocols, and attack types, generated using CICFlowMeter-V3.

### Data Processing
To ensure the quality and relevance of the data used for model training, the following steps were undertaken:

1. **Sampling**: Given the class imbalance within the dataset, stratified sampling was employed to maintain the distribution of categories. A balanced dataset was created by taking an equal number of 'Benign' and 'Attack' data points, which was crucial for training unbiased models.

2. **Feature Selection**: Based on my domain knowledge and feature correlation analysis, key features were selected for inclusion in the model training process. These features included network traffic descriptors like 'Flow Duration', 'Packet Length Mean', and 'Fwd IAT Mean', among others.

3. **Scaling**: Before feature selection and analysis, the data was scaled using standard scaling techniques. This was to ensure that the feature values had a standard normal distribution, which is beneficial for the performance of many machine learning algorithms.

### Model Training
Two machine learning models were trained to classify network traffic:

1. **Logistic Regression**: A baseline model that provides a reference for the performance of more complex models.
2. **XGBoost**: An advanced model that uses gradient boosting to provide improved classification accuracy.

### Evaluation
The models were rigorously evaluated using a variety of metrics, including precision-recall curves, ROC curves, and AUC scores, to ensure robustness and reliability in classifying network traffic as either 'Benign' or 'Attack'.


For a more detailed breakdown of the attacks and traffic analysis, refer to the CIC's dataset description at [UNB CIC Datasets](https://www.unb.ca/cic/datasets/ddos-2019.html).


## Results + charts


## Prerequisites
Make sure you have the following Python packages installed:
- pandas
- numpy
- xgboost
- scikit-learn
- matplotlib
- joblib

You can install these packages using pip:
```sh
pip install pandas numpy xgboost scikit-learn matplotlib joblib
```

## Installation
To get a local copy up and running, follow these steps:

1. Clone the repository:
```sh
git clone https://github.com/tnzmnjm/network-traffic-classifier.git
```
Navigate to the project directory:
```sh
cd network-traffic-classifier
```

## Usage
Run the EDA, training, and evaluation scripts provided in the repository. The process will train Logistic Regression and XGBoost models, evaluate their performance, and save the models to disk.

## Models
Logistic Regression: A baseline classifier with class weights balanced.
XGBoost: An implementation of the gradient boosting framework with a binary logistic objective.

## Evaluation
Model performance is evaluated using ROC curves, AUC scores, and Precision-Recall curves, available in the src/utils.py file.

## Citation
If you use the CICDDoS2019 dataset in your research, please include the following citation:
Iman Sharafaldin, Arash Habibi Lashkari, Saqib Hakak, and Ali A. Ghorbani, "Developing Realistic Distributed Denial of Service (DDoS) Attack Dataset and Taxonomy", IEEE 53rd International Carnahan Conference on Security Technology, Chennai, India, 2019.
Please see the paper [here](https://ieeexplore.ieee.org/abstract/document/8888419).

