import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.exceptions import FitFailedWarning
import warnings

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def load_dataset(file_path):
    return pd.read_csv(file_path)

def preprocess_features(df, feature_config):
    X = df.copy()
    transformers = []
    ohe_columns = []

    for feature_name, params in feature_config.items():
        if params['is_selected']:
            feature_type = params['feature_variable_type']
            details = params['feature_details']
            missing_values_strategy = details.get('missing_values', 'None')
            impute_with = details.get('impute_with', 'None')
            impute_value = details.get('impute_value', 0)

            if feature_type == 'numerical':
                if missing_values_strategy == 'Impute':
                    if impute_with == 'Average of values':
                        transformers.append((feature_name, SimpleImputer(strategy='mean'), [feature_name]))
                    elif impute_with == 'custom':
                        transformers.append((feature_name, SimpleImputer(strategy='constant', fill_value=impute_value), [feature_name]))
            elif feature_type == 'text':
                if missing_values_strategy == 'Impute':
                    transformers.append((feature_name, SimpleImputer(strategy='constant', fill_value='missing'), [feature_name]))
                ohe = OneHotEncoder(sparse=False)
                transformers.append((feature_name, ohe, [feature_name]))
                ohe_columns.append(feature_name)
    
    preprocessor = ColumnTransformer(transformers, remainder='passthrough')
    X_transformed = preprocessor.fit_transform(X)
    
    new_columns = []
    for feature_name, params in feature_config.items():
        if params['is_selected']:
            if params['feature_variable_type'] == 'text':
                ohe = preprocessor.named_transformers_[feature_name]
                categories = ohe.categories_[0]
                new_columns.extend([f"{feature_name}_{category}" for category in categories])
            else:
                new_columns.append(feature_name)
    
    return pd.DataFrame(X_transformed, columns=new_columns)

def generate_features(df, feature_generation_config):
    
    for interaction in feature_generation_config.get('linear_interactions', []):
        feature_name = f'{interaction[0]}_{interaction[1]}_interaction'
        df[feature_name] = df[interaction[0]] * df[interaction[1]]
    
    return df

def reduce_features(X, method, num_features):
    if method == 'PCA':
        pca = PCA(n_components=num_features)
        X_reduced = pca.fit_transform(X)
        return X_reduced
    else:
        return X

def get_target(df, target_column):
    return df[target_column]

def select_and_train_model(X, y, model_config):
    if model_config['RandomForestRegressor']['is_selected']:
        model = RandomForestRegressor()
        param_grid = {
            'n_estimators': range(model_config['RandomForestRegressor']['min_trees'], model_config['RandomForestRegressor']['max_trees'] + 1),
            'max_depth': range(model_config['RandomForestRegressor']['min_depth'], model_config['RandomForestRegressor']['max_depth'] + 1),
            'min_samples_leaf': range(model_config['RandomForestRegressor']['min_samples_per_leaf_min_value'], model_config['RandomForestRegressor']['min_samples_per_leaf_max_value'] + 1)
        }
        grid_search = GridSearchCV(model, param_grid, cv=5, error_score='raise')
        try:
            grid_search.fit(X, y)
        except ValueError as e:
            print(f"Error during model training: {e}")
            return None, None, None
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        return best_model, best_params, grid_search.cv_results_
    return None, None, None

def main(json_file_path):
    config = load_json(json_file_path)
    
    dataset_path = config['design_state_data']['session_info']['dataset']
    df = load_dataset(dataset_path)
    
    X = preprocess_features(df, config['design_state_data']['feature_handling'])
    
    X = generate_features(X, config['design_state_data']['feature_generation'])
    
    reduction_method = config['design_state_data']['feature_reduction']['feature_reduction_method']
    num_features_to_keep = int(config['design_state_data']['feature_reduction']['num_of_features_to_keep'])
    X = reduce_features(X, reduction_method, num_features_to_keep)
    
    target_column = config['design_state_data']['target']['target']
    y = get_target(df, target_column)
    
    if df[target_column].dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model, best_params, cv_results = select_and_train_model(X_train, y_train, config['design_state_data']['algorithms'])
    
    if model:
        print("Best Model:", model)
        
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print(f"R² Score: {r2:.4f}")
        print(f"Mean Squared Error: {mse:.4f}")
    else:
        print("Model training failed. Check the error messages for details.")

if __name__ == "__main__":
    json_file_path = "C:\\Users\\Bhumikka Pancharane\\Downloads\\DS_Assignment - internship\\Screening Test - DS\\data.json"
    main(json_file_path)
