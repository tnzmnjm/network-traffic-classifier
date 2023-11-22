import pandas as pd
from sklearn.linear_model import LogisticRegression
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

# Building a  Logistic Regression classifier. My dataset is balanced.
clf_logistic_regression = LogisticRegression(solver='saga',
                                             max_iter=100,
                                             class_weight='balanced')

# Fit the model to the training data
clf_logistic_regression.fit(train_features, target)

# To make sure my model is not overfitting, I will need to do the same
# evaluation on my training, validation, and test data
prediction_clf_logistic_regression_training = \
    clf_logistic_regression.predict(train_features)
evaluate_model(target, prediction_clf_logistic_regression_training,
               "Logistic Regression training ")


# Make predictions on the validation dataset and evaluate
prediction_clf_logistic_regression = \
    clf_logistic_regression.predict(validation_features)
evaluate_model(validation_true, prediction_clf_logistic_regression,
               "Logistic Regression validation")

# Make predictions on the test set and evaluate
test_predictions = clf_logistic_regression.predict(test_features)
evaluate_model(test_true, test_predictions, "Logistic Regression Test")


joblib.dump(clf_logistic_regression, 'models/saved_models/'
                                     'logistic_regression_model.joblib')

joblib.dump(test_true, 'data/predictions/y_true.joblib')
joblib.dump(test_predictions, 'data/predictions/'
                              'logistic_regression_predictions.joblib')
