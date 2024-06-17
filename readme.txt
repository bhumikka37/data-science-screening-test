# Data Science Screening Test

## Overview

This project involves preprocessing a dataset, generating new features, reducing the dimensionality of the features, and training a machine learning model using configurations specified in a JSON file.

## Steps Performed

### 1. Load JSON Configuration

The JSON configuration file is loaded to extract details about dataset paths, feature handling, feature generation, feature reduction, and model training.

### 2. Load Dataset

The dataset specified in the JSON configuration is loaded into a Pandas DataFrame.

### 3. Preprocess Features

Features are preprocessed according to the configuration:
- **Numerical Features**: Missing values are imputed with specified strategies (mean, custom value).
- **Categorical Features**: Missing values are imputed with 'missing', and features are one-hot encoded to convert them to a numerical format.

### 4. Feature Generation

New features are generated based on specified linear interactions. For example, a new feature can be created as the product of two existing features.

### 5. Feature Reduction

Dimensionality reduction is performed using PCA if specified. The number of features to keep is determined by the JSON configuration.

### 6. Get Target Variable

The target variable is extracted from the DataFrame based on the column specified in the JSON configuration. If the target variable is categorical, it is encoded using a LabelEncoder.

### 7. Split Data

The dataset is split into training and testing sets with a test size of 20%.

### 8. Select and Train Model

A machine learning model (RandomForestRegressor) is selected and trained using hyperparameters specified in the JSON configuration. GridSearchCV is used to find the best hyperparameters through cross-validation.

### 9. Model Evaluation

The model is evaluated on the test set using:
- **R² Score**: Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.
- **Mean Squared Error (MSE)**: Measures the average squared difference between the actual and predicted values.

### Error Handling and Debugging

- Added error handling to catch and print any issues during model training.
- Configured GridSearchCV to raise errors for better debugging.

## How to Run

1. Ensure you have Python and necessary libraries installed (`pandas`, `numpy`, `scikit-learn`).
2. Place your dataset and JSON configuration file in the specified paths.
3. Update the `json_file_path` in the script to point to your JSON configuration file.
4. Run the script using `python script_name.py`.

## Note

- Ensure that the JSON configuration is correctly formatted and contains all required fields.
- Adjust hyperparameters and configurations as necessary based on the error messages and output.

## Example Output