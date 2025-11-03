# Core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine Learning libraries
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                           balanced_accuracy_score, f1_score, precision_score, recall_score)
from sklearn.preprocessing import StandardScaler
import joblib


X_train = pd.read_csv('X_train.csv')

y_train = pd.read_csv('y_train.csv').squeeze()

onehot = joblib.load('onehot.pk1')
ordinalencoder = joblib.load('ordinalencoder.pk1')
labelencoder = joblib.load('labelencoder.pk1')

scaler = joblib.load('standardscaler.pk1')
selected_features = joblib.load('selectedfeatures.pk1')


baseline_model = LogisticRegression(
    random_state=234,
    max_iter=1000,
    class_weight='balanced'  # Handle class imbalance as recommended by EDA
)

# Train the model
baseline_model.fit(X_train, y_train)
