# Network Traffic Classifier

## Description
This project implements a network traffic classification system designed to identify different types of network attacks. The system includes Exploratory Data Analysis (EDA), pre-processing pipelines, Logistic Regression and XGBoost classification models, and evaluation metrics to assess performance.

## Dataset
The dataset used in this project is the CICDDoS2019, available on [Kaggle](https://www.kaggle.com/datasets/dhoogla/cicddos2019). The CICDDoS2019 dataset is created and maintained by the Canadian Institute for Cybersecurity (CIC). For more detailed feature descriptions, visit the [CIC's official dataset page](https://www.unb.ca/cic/datasets/ddos-2019.html).

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

