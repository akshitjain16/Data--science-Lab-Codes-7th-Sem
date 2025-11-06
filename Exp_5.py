# ------------------------------------------------------------
# Lab Task 1 : Sales Performance Report using Pandas
# ------------------------------------------------------------
# Objective:
# To perform data analysis on a company's sales data using Pandas
# Operations used: Augmentation, Aggregation, Pivoting, Mapping, Binning
# ------------------------------------------------------------

import pandas as pd
import numpy as np


# ----------------------- Step 2 -----------------------------
def create_dataset():
    sales_data = pd.DataFrame({
        'Region': ['North', 'South', 'East', 'West', 'North', 'South', 'East', 'West'],
        'Product': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'],
        'Units_Sold': [100, 150, 120, 130, 90, 110, 80, 140],
        'Unit_Price': [50, 45, 60, 55, 70, 65, 80, 75],
        'Category': [
            'Electronics', 'Electronics', 'Clothing', 'Clothing',
            'Grocery', 'Grocery', 'Grocery', 'Grocery'
        ]
    })

    print("📘 Initial Dataset:\n")
    print(sales_data)
    print("\n------------------------------------------------------------\n")
    return sales_data


# ----------------------- Step 3 -----------------------------
def augmentation(sales_data):
    print("1️⃣ Augmentation – Adding Derived Columns:\n")

    sales_data['Total_Revenue'] = sales_data['Units_Sold'] * sales_data['Unit_Price']
    sales_data['Discount (%)'] = sales_data['Total_Revenue'].apply(lambda x: 10 if x > 7000 else 5)
    sales_data['Final_Revenue'] = sales_data['Total_Revenue'] * (1 - sales_data['Discount (%)'] / 100)

    print(sales_data)
    print("\n------------------------------------------------------------\n")
    return sales_data


# ----------------------- Step 4 -----------------------------
def aggregation_report(sales_data):
    print("2️⃣ Aggregation – Regional Performance Summary:\n")

    region_summary = sales_data.groupby('Region').agg({
        'Units_Sold': 'sum',
        'Total_Revenue': ['sum', 'mean'],
        'Final_Revenue': 'sum'
    })

    print(region_summary)
    print("\n------------------------------------------------------------\n")


# ----------------------- Step 5 -----------------------------
def pivot_report(sales_data):
    print("3️⃣ Pivot Table – Product vs Region (Final Revenue):\n")

    pivot_table = sales_data.pivot(index='Product', columns='Region', values='Final_Revenue')
    print(pivot_table)
    print("\n------------------------------------------------------------\n")


# ----------------------- Step 6 -----------------------------
def mapping_profit_margin(sales_data):
    print("4️⃣ Mapping – Assigning Profit Margins:\n")

    profit_margin = {'Electronics': 15, 'Clothing': 20, 'Grocery': 10}
    sales_data['Profit_Margin (%)'] = sales_data['Category'].map(profit_margin)

    print(sales_data[['Product', 'Category', 'Profit_Margin (%)']])
    print("\n------------------------------------------------------------\n")


# ----------------------- Step 7 -----------------------------
def binning_revenue(sales_data):
    print("5️⃣ Binning – Categorizing Revenue Levels:\n")

    bins = [0, 5000, 9000, 13000]
    labels = ['Low', 'Medium', 'High']
    sales_data['Revenue_Level'] = pd.cut(sales_data['Final_Revenue'], bins=bins, labels=labels)

    print(sales_data[['Region', 'Product', 'Final_Revenue', 'Revenue_Level']])
    print("\n------------------------------------------------------------\n")


# ----------------------- Step 8 -----------------------------
def final_dataset_display(sales_data):
    print("6️⃣ Final Sales Performance Dataset:\n")
    print(sales_data)
    print("\n------------------------------------------------------------\n")


# ----------------------- Step 9 -----------------------------
def insights_summary():
    print("📊 Insights Summary:")
    print("- Most regions fall into the 'Medium' revenue category.")
    print("- The West region generated the highest revenue.")
    print("- Electronics and Clothing products have higher profit margins.")
    print("- Augmentation, aggregation, pivoting, and binning provided deeper insights.")
    print("\n✅ Lab Task 1 Completed Successfully!")


# ----------------------- MAIN ------------------------------
def main():
    sales_data = create_dataset()
    sales_data = augmentation(sales_data)
    aggregation_report(sales_data)
    pivot_report(sales_data)
    mapping_profit_margin(sales_data)
    binning_revenue(sales_data)
    final_dataset_display(sales_data)
    insights_summary()


if __name__ == "__main__":
    main()
