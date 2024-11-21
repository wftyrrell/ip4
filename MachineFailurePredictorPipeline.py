import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# (3). Filter six attributes
def filterData():
    """
    Reads the CSV file and filters the required attributes.

    Returns:
    pd.DataFrame: Filtered data.
    """
    try:
        data = pd.read_csv('ai4i2020.csv')
        selectedData = data[['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 
                          'Torque [Nm]', 'Tool wear [min]', 'Machine failure']]
        return selectedData
    except FileNotFoundError as e:
        logging.error("File not found: %s", e)
        return None

# (4). Data preprocessing 
def preprocessData():
    
    selectedData = filterData()
    encoder = LabelEncoder()
    scaler = StandardScaler()
    selectedData['Type'] = encoder.fit_transform(selectedData['Type'])  
    scaledData = scaler.fit_transform(selectedData.drop('Machine failure', axis=1))
    return pd.DataFrame(scaledData, columns=selectedData.drop('Machine failure', axis=1).columns)

# (5). RandomUnderSampler data balancing
def dataBalance():
    """
    Balances the dataset using RandomUnderSampler to address class imbalance.

    Returns:
    pd.DataFrame: Balanced dataset with target and features.
    """
    selectedData = filterData()
    scaledData = preprocessData()
    X = scaledData
    y = selectedData['Machine failure']
    rus = RandomUnderSampler(sampling_strategy={0: 339, 1: 339}, random_state=42)
    X_res, y_res = rus.fit_resample(X, y)
    balancedData = pd.DataFrame(X_res, columns=scaledData.columns)
    balancedData['Machine failure'] = y_res
    return balancedData

# (6). 5-fold cross-validation 
def crossValidation(X_train, y_train):
    """"
    Defines hyperparameter grids for various machine learning models.

    Parameters:
    X_train (pd.DataFrame): Training data features.
    y_train (pd.Series): Training data labels.
    """
    param_grids = {
        'MLP': {
            'hidden_layer_sizes': [(50,), (100,), (100, 50), (150,)],
            'activation': ['relu', 'tanh'],
            'learning_rate': ['constant', 'adaptive'],
            'max_iter': [2000],
            'early_stopping': [True],
        },
        'SVC': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        },
        'KNN': {
            'n_neighbors': [3, 5, 7, 10],
            'p': [1, 2],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        },
        'DecisionTree': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 20],
            'ccp_alpha': [0.0, 0.1, 0.2]
        },
        'LogisticRegression': {
            'penalty': ['l2'],  
            'max_iter': [5000],
            'solver': ['lbfgs', 'liblinear']  
        }
    }

    models = {
        'MLP': MLPClassifier(max_iter=2000, early_stopping=True),
        'SVC': SVC(),
        'KNN': KNeighborsClassifier(),
        'DecisionTree': DecisionTreeClassifier(),
        'LogisticRegression': LogisticRegression()
    }

    results = []
    bestParameters = {}

    for modelName in models:
        print(f"Training {modelName} model...")
        grid_search = GridSearchCV(models[modelName], param_grids[modelName], cv=5, scoring=make_scorer(matthews_corrcoef))
        grid_search.fit(X_train, y_train)

        bestParameters[modelName] = grid_search.best_params_
        best_score = grid_search.best_score_

        results.append([modelName, bestParameters[modelName], best_score])

    results_df = pd.DataFrame(results, columns=["ML Trained Model", "Best Set of Parameter Values", "MCC Score (5-Fold CV)"])
    results_df.to_csv("Table_1.csv", index=False)
    print("\n" + str(results_df))

    return bestParameters

# (7) Evaluate models on the test set
def evaluate_models(X_train, X_test, y_train, y_test, models, best_params):
    """
    Evaluates models using the provided training and test data.

    Parameters:
    X_train (pd.DataFrame): Training data features.
    X_test (pd.DataFrame): Test data features.
    y_train (pd.Series): Training data labels.
    y_test (pd.Series): Test data labels.
    models (dict): Dictionary of models to evaluate.
    param_grids (dict): Dictionary of hyperparameter grids for each model.
    """
    testResults = []

    for model_name in models:
        print(f"Evaluating {model_name} model on test set...")
        
        model = models[model_name].set_params(**best_params[model_name])
        
        model.fit(X_train, y_train)
        
        y_predict = model.predict(X_test)
        mcc_score = matthews_corrcoef(y_test, y_predict)
        
        testResults.append([model_name, best_params[model_name], mcc_score])
    
    results_df = pd.DataFrame(testResults, columns=["ML Trained Model", "Best Set of Parameter Values", "MCC-score on Testing Data (20%)"])
    results_df.to_csv("Table_2.csv", index=False)
    print("\n" + str(results_df))
    
    best_model_row = results_df.loc[results_df['MCC-score on Testing Data (20%)'].idxmax()]
    print(f"\nThe model with the highest MCC score is: {best_model_row['ML Trained Model']} with a score of {best_model_row['MCC-score on Testing Data (20%)']}")

def main():
    # (3). Filter six attributes
    print("Part 3: Filtered Data")
    filtered_data = filterData()
    print(filtered_data.head())  
    print("\n")

    # (4). Preprocessed Data
    print("Part 4: Preprocessed Data")
    preprocessed_data = preprocessData()
    print(preprocessed_data.head())  
    print("\n")

    # (5) Balanced Data
    print("Part 5: Balanced Data")
    balanced_data = dataBalance()
    print(balanced_data.head()) 
    print("\n")

    # (6) Cross-validation for model tuning
    print("Part 6: Cross-validation for Model Tuning")
    X = balanced_data.drop('Machine failure', axis=1)
    y = balanced_data['Machine failure']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    best_params = crossValidation(X_train, y_train)
    
    print("\n")

    # (7). Model Evaluation
    print("Part 7: Model Evaluation on the Test Set")
    models = {
        'MLP': MLPClassifier(),
        'SVC': SVC(),
        'KNN': KNeighborsClassifier(),
        'DecisionTree': DecisionTreeClassifier(),
        'LogisticRegression': LogisticRegression()
    }

    evaluate_models(X_train, X_test, y_train, y_test, models, best_params)
    
if __name__ == '__main__':
    main()
