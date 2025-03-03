import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
import seaborn as sns
import matplotlib.pyplot as plt

# Custom transformer to compute the median over specified chunk sizes
class ChunkedMedianTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, chunk_size=5):
        self.chunk_size = chunk_size
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        n_chunks = len(X) // self.chunk_size
        return np.array([np.median(X[i*self.chunk_size:(i+1)*self.chunk_size]) for i in range(n_chunks)])

# Function to load data and preprocess it
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df1 = df.drop(['timestamp', 'Unnamed: 0'], axis=1)  # Removing unnecessary columns
    return df1

# Function to train multiple classifiers
def train_models(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Standardizing features
    
    # Initializing models
    log_reg = LogisticRegression()
    svc = SVC()
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Training models
    log_reg.fit(X_train_scaled, y_train)
    svc.fit(X_train_scaled, y_train)
    rf.fit(X_train_scaled, y_train)
    
    # Voting classifier (combining multiple models)
    voting_clf = VotingClassifier(estimators=[
        ('log_reg', log_reg),
        ('rf', rf)
    ], voting='hard')
    voting_clf.fit(X_train_scaled, y_train)
    
    return scaler, log_reg, svc, rf, voting_clf

# Function to evaluate models
def evaluate_models(models, X_test, y_test):
    scaler, log_reg, svc, rf, voting_clf = models
    X_test_scaled = scaler.transform(X_test)
    
    # Cross-validation scores
    scores = {
        'Logistic Regression': log_reg.score(X_test_scaled, y_test),
        'SVM': svc.score(X_test_scaled, y_test),
        'Random Forest': rf.score(X_test_scaled, y_test),
        'Voting Classifier': voting_clf.score(X_test_scaled, y_test)
    }
    
    return scores

# Function to monitor file for new data and make predictions
def monitor_and_process_file(filepath, models):
    scaler, log_reg, svc, rf, voting_clf = models
    
    while True:
        df = load_and_preprocess_data(filepath)
        if df is not None and not df.empty:
            X_new = df.iloc[:, :-1]  # Assuming last column is the label
            X_new_scaled = scaler.transform(X_new)
            predictions = voting_clf.predict(X_new_scaled)
            print("Predictions:", predictions)
       

# Main execution block
if __name__ == "__main__":
    filepath = "../Lab_1/emg_filename.csv"  # Change to your actual file path
    df = load_and_preprocess_data(filepath)
    
    X = df.iloc[:, :-1]  # Features
    y = df.iloc[:, -1]   # Labels
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = train_models(X_train, y_train)
    scores = evaluate_models(models, X_test, y_test)
    
    print("Model Evaluation Scores:", scores)
    
    # Start monitoring the file in real-time (optional)
    # monitor_and_process_file(filepath, models)

    # Plot histogram of feature distribution
    sns.histplot(X_train, bins=30, kde=True)
    plt.title("Feature Distribution")
    plt.show()


