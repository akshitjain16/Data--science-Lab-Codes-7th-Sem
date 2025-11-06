# ------------------------------------------------------------
# Experiment 4 (Alternate Version)
# Aim: Read data from text/CSV and Web, and perform descriptive analytics
# Dataset: Iris
# ------------------------------------------------------------

import pandas as pd


# ----------------------- Step 1 -----------------------------
def read_csv_data():
    print("1️⃣ Reading data from a text/CSV file:\n")

    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    iris = pd.read_csv(url)

    print("First 5 rows of the dataset:\n", iris.head(), "\n")
    return iris


# ----------------------- Step 2 -----------------------------
def dataset_info(iris):
    print("2️⃣ Dataset Information:\n")
    print(iris.info(), "\n")

    print("Shape of dataset:", iris.shape)
    print("Column names:", iris.columns.tolist(), "\n")


# ----------------------- Step 3 -----------------------------
def descriptive_stats(iris):
    print("3️⃣ Descriptive Statistics:\n")
    print(iris.describe(), "\n")

    print("4️⃣ Checking for Missing Values:\n")
    print(iris.isnull().sum(), "\n")

    print("5️⃣ Mean, Median, and Mode of Numeric Columns:\n")
    print("Mean:\n", iris.mean(numeric_only=True))
    print("\nMedian:\n", iris.median(numeric_only=True))
    print("\nMode:\n", iris.mode().iloc[0], "\n")


# ----------------------- Step 4 -----------------------------
def grouping_aggregation(iris):
    print("6️⃣ Grouping by Species and Calculating Mean:\n")
    print(iris.groupby("species").mean(numeric_only=True), "\n")


# ----------------------- Step 5 -----------------------------
def correlation_analysis(iris):
    print("7️⃣ Correlation Matrix:\n")
    print(iris.corr(numeric_only=True), "\n")


# ----------------------- Step 6 -----------------------------
def export_summary(iris):
    summary = iris.describe()
    summary.to_csv("iris_summary.csv", index=True)
    print("✅ Descriptive summary exported successfully as 'iris_summary.csv'!")


# ----------------------- MAIN ------------------------------
def main():
    iris = read_csv_data()
    dataset_info(iris)
    descriptive_stats(iris)
    grouping_aggregation(iris)
    correlation_analysis(iris)
    export_summary(iris)
    print("\n🎯 Experiment Completed Successfully!")


if __name__ == "__main__":
    main()
