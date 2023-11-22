import pandas as pd
import numpy as np
import xgboost as xgb
from src import evaluate_model
import joblib


df_train = pd.read_csv('data/train_df.csv')
df_validation = pd.read_csv('data/validation_df.csv')
df_test = pd.read_csv('data/test_df.csv')


feature_columns = ['Protocol', 'Fwd Packet Length Mean', 'Flow IAT Mean',
                   'Flow Bytes/s', 'Packet Length Mean',
                   'Packet Length Min', 'Avg Fwd Segment Size',
                   'Avg Packet Size', 'Subflow Fwd Bytes',
                   'Fwd Act Data Packets']


train_features = df_train[feature_columns]
target = df_train['Class']

validation_features = df_validation[feature_columns]
validation_true = df_validation['Class']

test_features = df_test[feature_columns]
test_true = df_test['Class']


clf_xgb = xgb.XGBClassifier(n_estimators=6,
                            max_depth=5,
                            learning_rate=1,
                            objective='binary:logistic')

# Fit the model to the training data
clf_xgb.fit(train_features, target)


# To make sure my model is not overfitting, I will need to do the same
# evaluation on my training, validation, and test data
prediction_clf_xgb_training = \
    clf_xgb.predict(train_features)
evaluate_model(target, prediction_clf_xgb_training,
               "XGBoost training ")


# Make predictions on the validation dataset and evaluate
validation_prediction = clf_xgb.predict(validation_features)
evaluate_model(validation_true, validation_prediction,
               "XGBoost Validation")

# Make predictions on the test set and evaluate
test_predictions = clf_xgb.predict(test_features)
evaluate_model(test_true, test_predictions, "XGBoost Test")


joblib.dump(clf_xgb, 'models/saved_models/xgb_model.joblib')
joblib.dump(test_predictions, 'data/predictions/xgboost_predictions.joblib')