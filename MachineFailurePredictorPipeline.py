
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.metrics import matthews_corrcoef


def preprocess_data(data):
    # Separate features and target
    # Separate features and target
    X = data[['Type', 'Air temperature [K]', 'Process temperature [K]', 
              'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
    y = data['Machine failure']

    # One-Hot Encode the "Type" column
    encoder = OneHotEncoder(sparse_output=False, drop='first')  # Updated parameter name
    type_encoded = encoder.fit_transform(X[['Type']])
    type_encoded_df = pd.DataFrame(type_encoded, columns=encoder.get_feature_names_out(['Type']))

    # Drop the original "Type" column and concatenate encoded columns
    X = X.drop(columns=['Type']).reset_index(drop=True)
    X = pd.concat([X, type_encoded_df], axis=1)

    return X, y


def main():
    
    
    data = pd.read_csv("ai4i2020.csv")
    X, y = preprocess_data(data)
    print("Preprocessed Features:\n", X.head())
    print("Target Variable:\n", y.head())
  
    data = pd.get_dummies(data, columns=['Type'], drop_first=True)
    
    scaler = StandardScaler()
    numeric_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    data[numeric_features] = scaler.fit_transform(data[numeric_features])
    
    X = data.drop(columns=['Machine failure'])
    y = data['Machine failure']
    sampler = RandomUnderSampler()
    X_resampled, y_resampled = sampler.fit_resample(X, y)   
    
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    scorer = make_scorer(matthews_corrcoef)

    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        'learning_rate': ['constant', 'adaptive']
    }
    grid_search = GridSearchCV(MLPClassifier(max_iter=1000), param_grid, scoring=scorer, cv=5)
    grid_search.fit(X_train, y_train)

    print("Best parameters:", grid_search.best_params_)
    print("MCC score:", grid_search.best_score_)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    mcc = matthews_corrcoef(y_test, y_pred)
    print("Test MCC:", mcc)

    # Step 1: Load and preprocess data
    # Step 2: Balance data using RandomUnderSampler
    # Step 3: Split data into training and testing sets
    # Step 4: Train and fine-tune models using cross-validation
    # Step 5: Evaluate models on the test set
    # Step 6: Output Table 1 and Table 2

if __name__ == "__main__":
    main()