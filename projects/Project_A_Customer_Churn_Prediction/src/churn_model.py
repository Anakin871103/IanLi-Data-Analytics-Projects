# churn_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

class ChurnModel:
    def __init__(self, data):
        self.data = data
        self.model = RandomForestClassifier()

    def preprocess_data(self):
        # Example preprocessing steps
        self.data.fillna(0, inplace=True)
        self.data = pd.get_dummies(self.data)

    def train_model(self):
        X = self.data.drop('churn', axis=1)
        y = self.data['churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        
        print(confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions))

    def predict(self, new_data):
        return self.model.predict(new_data)