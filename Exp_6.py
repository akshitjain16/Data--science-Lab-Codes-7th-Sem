# ------------------------------------------------------------
# LAB TASK 2: Chart Plots using Matplotlib and Seaborn
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ----------------------- Step 2 -----------------------------
def create_dataset():
    df = pd.DataFrame({
        'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        'Sales': [200, 240, 230, 300, 310, 400, 370, 380, 410, 450, 470, 500],
        'Profit': [20, 30, 25, 40, 45, 60, 55, 58, 65, 70, 72, 80],
        'Region': ['North', 'South', 'East', 'West'] * 3
    })

    print("✅ Sample Dataset:\n")
    print(df.head())
    print("\n------------------------------------------------------------\n")

    return df


# ----------------------- Step 3 -----------------------------
def bar_chart(df):
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Month', y='Sales', data=df, color='skyblue', edgecolor='black')
    plt.title("Monthly Sales (Bar Chart)")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.show()


# ----------------------- Step 4 -----------------------------
def histogram(df):
    plt.figure(figsize=(8, 5))
    plt.hist(df['Sales'], bins=8, color='orange', edgecolor='black')
    plt.title("Sales Distribution (Histogram)")
    plt.xlabel("Sales")
    plt.ylabel("Frequency")
    plt.show()


# ----------------------- Step 5 -----------------------------
def line_chart(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df['Month'], df['Sales'], marker='o', label='Sales', color='blue')
    plt.plot(df['Month'], df['Profit'], marker='s', label='Profit', color='green')
    plt.title("Sales and Profit Trend (Line Chart)")
    plt.xlabel("Month")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()


# ----------------------- Step 6 -----------------------------
def area_chart(df):
    plt.figure(figsize=(10, 5))
    plt.fill_between(df['Month'], df['Sales'], color='skyblue', alpha=0.5)
    plt.title("Sales Over Time (Area Chart)")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.show()


# ----------------------- Step 7 -----------------------------
def pie_chart(df):
    region_sales = df.groupby('Region')['Sales'].sum()
    plt.figure(figsize=(6, 6))
    plt.pie(region_sales, labels=region_sales.index, autopct='%1.1f%%', startangle=90)
    plt.title("Sales by Region (Pie Chart)")
    plt.show()


# ----------------------- Step 8 -----------------------------
def scatter_plot(df):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='Sales', y='Profit', data=df, color='purple', s=80)
    plt.title("Sales vs Profit (Scatter Plot)")
    plt.xlabel("Sales")
    plt.ylabel("Profit")
    plt.show()


# ----------------------- Step 9 -----------------------------
def box_plot(df):
    plt.figure(figsize=(8, 5))
    sns.boxplot(y='Profit', data=df, color='lightgreen')
    plt.title("Profit Distribution (Box Plot)")
    plt.ylabel("Profit")
    plt.show()


# ----------------------- Step 10 -----------------------------
def pareto_chart(df):
    df_sorted = df.sort_values(by='Sales', ascending=False)
    df_sorted['CumPerc'] = df_sorted['Sales'].cumsum() / df_sorted['Sales'].sum() * 100

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(df_sorted['Month'], df_sorted['Sales'], color='skyblue')
    ax2 = ax1.twinx()
    ax2.plot(df_sorted['Month'], df_sorted['CumPerc'], color='red', marker='o')

    ax1.set_title("Pareto Chart - Sales Analysis")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Sales")
    ax2.set_ylabel("Cumulative %")
    plt.show()


# ----------------------- MAIN ------------------------------
def main():
    # Global style
    sns.set(style="whitegrid")
    plt.rcParams["figure.figsize"] = (8, 5)

    df = create_dataset()
    bar_chart(df)
    histogram(df)
    line_chart(df)
    area_chart(df)
    pie_chart(df)
    scatter_plot(df)
    box_plot(df)
    pareto_chart(df)

    print("\n✅ All charts executed successfully!")


if __name__ == "__main__":
    main()
