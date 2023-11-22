This notebook is used for conducting an exploratory data analysis (EDA) on network traffic data, with the aim of classifying the traffic into benign and malicious (attack) packets.

The code includes functions to load and clean data, encode the 'Class' column, check for class imbalances, visualize feature correlations and scales, balance the dataset, and split it into train, validation, and test sets. It also scales the features to prepare the data for machine learning model training.

If you execute 'Run All', the code will process the dataset through all the steps mentioned above and will produce a balanced and preprocessed dataset ready for model training and save the processed datasets and the scaler to disk.

The notebook serves as a comprehensive preprocessing tool that transforms raw network traffic data into a format suitable for training machine learning models, ensuring that the input data is clean, balanced, and standardized to improve model accuracy and performance.