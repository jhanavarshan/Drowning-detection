import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

class DrowningDetectionModel:
    def __init__(self):
        self.model = None
        self.label_mapping = {
            'Normal': 0,
            'Abnormal Swim-Angle': 1,
            'Abnormal Heart-Rate': 2,
            'Abnormal Humidity': 3,
            'Abnormal Temperature': 4
        }
        self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
        self.model_path = 'drowning_detection_model.pkl'
        self.label_mapping_path = 'label_mapping.pkl'
        
    def load_data(self):
        """Load and preprocess the dataset"""
        df = pd.read_csv("Swim_Data.csv")
        df['Label_encoded'] = df['Label'].map(self.label_mapping)
        X = df[['Temperature', 'Humidity', 'Heart_rate', 'Swim_angle']]
        y = df['Label_encoded']
        return X, y
    
    def train(self):
        """Train the model and save it"""
        X, y = self.load_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        
        self.save_model()
    
    def save_model(self):
        """Save the trained model and label mapping"""
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.label_mapping, self.label_mapping_path)
    
    def load_model(self):
        """Load the saved model"""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            self.label_mapping = joblib.load(self.label_mapping_path)
            self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
            return True
        return False
    
    def predict(self, temperature, humidity, heart_rate, swim_angle):
        """Make a prediction"""
        if not self.model:
            if not self.load_model():
                self.train()
        
        features = np.array([[temperature, humidity, heart_rate, swim_angle]])
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        
        return {
            'prediction': self.reverse_label_mapping[prediction],
            'probability': float(max(probability)),
            'details': {
                'Normal': float(probability[self.label_mapping['Normal']]),
                'Abnormal Swim-Angle': float(probability[self.label_mapping['Abnormal Swim-Angle']]),
                'Abnormal Heart-Rate': float(probability[self.label_mapping['Abnormal Heart-Rate']]),
                'Abnormal Humidity': float(probability[self.label_mapping['Abnormal Humidity']]),
                'Abnormal Temperature': float(probability[self.label_mapping['Abnormal Temperature']])
            }
        }