""" 
Group C: Assignment No. 11
Assignment Title: How to Train a Neural Network with
TensorFlow/Pytorch and evaluation of logistic regression using
tensorflow
"""
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import torch
import torch.nn as nn
import torch.optim as optim

def load_data():
    # Load the breast cancer dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    return X, y

def preprocess_data(X, y):
    # Preprocess the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, y_train):
    # Train a Logistic Regression Model using TensorFlow
    logistic_model = Sequential([
        Dense(1, input_dim=X_train.shape[1], activation='sigmoid')
    ])
    logistic_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    logistic_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    return logistic_model

def train_neural_network(X_train, y_train):
    # Train a Neural Network Model using TensorFlow
    nn_model = Sequential([
        Dense(64, input_dim=X_train.shape[1], activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    nn_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    return nn_model

def evaluate_model(model, X_test, y_test):
    # Model Evaluation
    y_pred = (model.predict(X_test) > 0.5).astype("int32").flatten()
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    return acc, report, auc

def display_results(acc, report, auc):
    print("Accuracy:", acc)
    print("Classification Report:\n", report)
    print("ROC AUC Score:", auc)

# Main menu
def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    while True:
        print("\nMenu:")
        print("1. Train Logistic Regression Model")
        print("2. Train Neural Network Model")
        print("3. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            logistic_model = train_logistic_regression(X_train, y_train)
            acc, report, auc = evaluate_model(logistic_model, X_test, y_test)
            print("\nLogistic Regression Model:")
            display_results(acc, report, auc)
        elif choice == '2':
            nn_model = train_neural_network(X_train, y_train)
            acc, report, auc = evaluate_model(nn_model, X_test, y_test)
            print("\nNeural Network Model:")
            display_results(acc, report, auc)
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a valid option.")

if __name__ == "__main__":
    main()
