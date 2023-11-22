from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import joblib


y_true = joblib.load('data/predictions/y_true.joblib')
logistic_regression_predictions = joblib.load('data/predictions/logistic_regression_predictions.joblib')
xgboost_predictions = joblib.load('data/predictions/xgboost_predictions.joblib')

# Calculate ROC curve and ROC area for each model
fpr1, tpr1, _ = roc_curve(y_true, logistic_regression_predictions)
roc_auc1 = auc(fpr1, tpr1)

fpr2, tpr2, _ = roc_curve(y_true, xgboost_predictions)
roc_auc2 = auc(fpr2, tpr2)

# Plot ROC curve
plt.figure()
lw = 2 # Line width
plt.plot(fpr1, tpr1, color='orange', lw=lw, label='Logistic Regression (AUC = %0.2f)' % roc_auc1)
plt.plot(fpr2, tpr2, color='blue', lw=lw, label='XGBoost (AUC = %0.2f)' % roc_auc2)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Example')
plt.legend(loc="lower right")
plt.show()


# Precision-recall curves are indeed more informative than ROC curves in situations
# where there is a significant class imbalance.

# Calculate precision-recall curve
precision1, recall1, _ = precision_recall_curve(y_true, logistic_regression_predictions)
precision2, recall2, _ = precision_recall_curve(y_true, xgboost_predictions)

# Calculate average precision
# Average Precision (AP): A single summary figure that accounts for precision at each threshold, weighted by the increase in recall from the previous threshold.
average_precision1 = average_precision_score(y_true, logistic_regression_predictions)
average_precision2 = average_precision_score(y_true, xgboost_predictions)

# Plot precision-recall curve
plt.figure()

plt.plot(recall1, precision1, label='Logistic Regression (AP = {:.2f})'.format(average_precision1))
plt.plot(recall2, precision2, label='XGBoost (AP = {:.2f})'.format(average_precision2))

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve')
plt.legend(loc="best")
plt.show()
