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


## Results and Analysis

The project aimed to develop machine learning models to distinguish between benign and DDoS attack traffic within network data. The Logistic Regression and XGBoost models were trained and tested on the CICDDoS2019 dataset, and their performance was evaluated based on accuracy, precision, recall, and AUC scores.

### Logistic Regression
- **Training**: Accuracy: 94.16%, Precision: 94.31%, Recall: 94.16%, AUC: 94.16%
- **Validation**: Accuracy: 94.40%, Precision: 94.50%, Recall: 94.40%, AUC: 94.40%
- **Test**: Accuracy: 82.46%, Precision: 89.98%, Recall: 82.46%, AUC: 88.42%

The Logistic Regression model demonstrated high effectiveness on the training and validation datasets. However, the test recall indicates that the model may miss a higher proportion of positive cases (attacks) when subjected to new, unseen data.

### XGBoost
- **Training**: Accuracy: 99.36%, Precision: 99.36%, Recall: 99.36%, AUC: 99.36%
- **Validation**: Accuracy: 96.10%, Precision: 96.30%, Recall: 96.10%, AUC: 96.10%
- **Test**: Accuracy: 81.28%, Precision: 89.85%, Recall: 81.28%, AUC: 87.96%

XGBoost outperformed Logistic Regression in training and validation. However, it too experienced a drop in recall on the test data, suggesting challenges in consistently identifying attack instances across different data segments.

### Evaluation Charts
Performance of the models is further detailed in the charts below, illustrating the trade-offs between various evaluation metrics.

![ROC Curve](path/to/roc_curve.png)
*ROC curves showing the true positive rate against the false positive rate for both models.*

![Precision-Recall Curve](path/to/precision_recall_curve.png)
*Precision-Recall curves highlighting the balance between precision and recall in model predictions.*

### Discussion
The reduction in recall on the test set for both models is a critical observation, particularly in the field of cybersecurity where failing to detect an attack can have significant consequences. It suggests a need for further model refinement, perhaps through additional feature engineering, data augmentation, or advanced model tuning, to improve the models' ability to generalize to new data.

Both models showed a promising ability to classify network traffic, with XGBoost displaying a particularly strong capability to learn complex patterns. Nonetheless, the drop in test recall underscores the importance of continuous model evaluation and updating to maintain high performance as new attack vectors emerge.


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

