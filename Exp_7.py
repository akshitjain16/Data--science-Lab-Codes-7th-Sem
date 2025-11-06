# ------------------------------------------------------------
# LAB TASK 3 : Titanic Survival Prediction using Logistic Regression
# ------------------------------------------------------------

# Step 1: Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sns.set(style="whitegrid")


# ----------------------- Step 2 -----------------------------
def load_dataset():
    titanic = sns.load_dataset('titanic')
    print("✅ Titanic Dataset Loaded Successfully!\n")
    print(titanic.head())
    print("\n------------------------------------------------------------\n")
    return titanic


# ----------------------- Step 3 -----------------------------
def preprocess_data(titanic):
    print("🔍 Checking Missing Values:\n")
    print(titanic.isnull().sum())

    titanic = titanic[['survived', 'pclass', 'sex', 'age', 'fare', 'embarked']]
    titanic.dropna(inplace=True)

    titanic['sex'] = titanic['sex'].map({'male': 0, 'female': 1})
    titanic['embarked'] = titanic['embarked'].map({'C': 0, 'Q': 1, 'S': 2})

    print("\n✅ Cleaned and Encoded Dataset:\n")
    print(titanic.head())
    print("\n------------------------------------------------------------\n")

    return titanic


# ----------------------- Step 4 -----------------------------
def perform_eda(titanic):
    plt.figure(figsize=(6, 4))
    sns.countplot(x='survived', data=titanic, palette='Set2')
    plt.title("Survival Count")
    plt.xlabel("Survived (0 = No, 1 = Yes)")
    plt.ylabel("Count")
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.barplot(x='sex', y='survived', data=titanic, palette='coolwarm')
    plt.title("Survival Rate by Gender")
    plt.xlabel("Gender (0=Male, 1=Female)")
    plt.ylabel("Survival Rate")
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.barplot(x='pclass', y='survived', data=titanic, palette='mako')
    plt.title("Survival Rate by Passenger Class")
    plt.xlabel("Passenger Class")
    plt.ylabel("Survival Rate")
    plt.show()


# ----------------------- Step 5 -----------------------------
def split_data(titanic):
    X = titanic.drop('survived', axis=1)
    y = titanic['survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("✅ Data Split Completed:")
    print(f"Training Samples: {X_train.shape[0]} | Testing Samples: {X_test.shape[0]}")
    print("\n------------------------------------------------------------\n")

    return X_train, X_test, y_train, y_test


# ----------------------- Step 6 -----------------------------
def build_model(X_train, y_train):
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model


# ----------------------- Step 7 -----------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"🎯 Model Accuracy: {accuracy * 100:.2f}%\n")
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return cm


# ----------------------- Step 8 -----------------------------
def plot_confusion_matrix(cm):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Confusion Matrix - Titanic Survival Prediction")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# ----------------------- MAIN ------------------------------
def main():
    titanic = load_dataset()
    titanic = preprocess_data(titanic)
    perform_eda(titanic)
    X_train, X_test, y_train, y_test = split_data(titanic)
    model = build_model(X_train, y_train)
    cm = evaluate_model(model, X_test, y_test)
    plot_confusion_matrix(cm)
    print("\n✅ Titanic Survival Prediction completed successfully!")


if __name__ == "__main__":
    main()
