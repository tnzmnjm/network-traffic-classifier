from sklearn.metrics import accuracy_score, precision_score, \
    recall_score, roc_auc_score


def evaluate_model(true_values, predictions, model_name):
    accuracy = accuracy_score(true_values, predictions)
    precision = precision_score(true_values, predictions, average='weighted')
    recall = recall_score(true_values, predictions, average='weighted')
    auc = roc_auc_score(true_values, predictions, average='weighted')
    print('*****')
    print(f"{model_name} accuracy: {accuracy:.4f}")
    print(f"{model_name} precision: {precision:.4f}")
    print(f"{model_name} recall: {recall:.4f}")
    print(f"{model_name} AUC: {auc:.4f}")
    print('*****')
