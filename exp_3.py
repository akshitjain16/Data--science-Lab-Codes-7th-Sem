# ------------------------------------------------------------
# Experiment 3 : Working with Pandas DataFrames
# Aim: To create, manipulate, and perform operations on DataFrames
# ------------------------------------------------------------

import pandas as pd


# ----------------------- Step 2 -----------------------------
def create_dataframe():
    print("1️⃣ Creating DataFrame:\n")

    data = {
        'Name': ['Akshit', 'Riya', 'Sanjay', 'Aman', 'Vedant', 'Ria'],
        'Age': [22, 25, 23, 24, 26, 21],
        'Marks': [88, 92, 79, 85, 90, 87],
        'City': ['Delhi', 'Mumbai', 'Pune', 'Chennai', 'Bangalore', 'Kolkata']
    }

    df = pd.DataFrame(data)
    print(df)
    return df


# ----------------------- Step 3 -----------------------------
def view_dataframe_info(df):
    print("\n2️⃣ Viewing DataFrame Information:\n")
    print("Head of DataFrame:\n", df.head())
    print("\nTail of DataFrame:\n", df.tail())
    print("\nShape of DataFrame:", df.shape)
    print("\nData Types:\n", df.dtypes)
    print("\nInfo of DataFrame:")
    print(df.info())


# ----------------------- Step 4 -----------------------------
def access_rows_columns(df):
    print("\n3️⃣ Accessing Columns and Rows:\n")
    print("Accessing 'Name' column:\n", df['Name'])
    print("\nAccessing first two rows using head():\n", df.head(2))
    print("\nAccessing by index using iloc:\n", df.iloc[1:4])


# ----------------------- Step 5 -----------------------------
def add_new_column(df):
    print("\n4️⃣ Adding a new column (Grade):\n")
    df['Grade'] = ['A', 'A+', 'B', 'B+', 'A', 'A-']
    print(df)


# ----------------------- Step 6 -----------------------------
def modify_values(df):
    print("\n5️⃣ Modifying Data:\n")
    df.loc[2, 'Marks'] = 82  # Changing Sanjay's Marks
    print("Updated DataFrame:\n", df)


# ----------------------- Step 7 -----------------------------
def filter_and_sort(df):
    print("\n6️⃣ Filtering and Sorting:\n")
    print("Students with Marks > 85:\n", df[df['Marks'] > 85])
    print("\nDataFrame sorted by Age:\n", df.sort_values(by='Age'))


# ----------------------- Step 8 -----------------------------
def describe_stats(df):
    print("\n7️⃣ Descriptive Statistics:\n")
    print(df.describe())


# ----------------------- Step 9 -----------------------------
def delete_column(df):
    print("\n8️⃣ Deleting the 'City' column:\n")
    df = df.drop('City', axis=1)
    print(df)
    return df


# ----------------------- Step 10 -----------------------------
def export_to_csv(df):
    print("\n9️⃣ Exporting to CSV file (optional):")
    df.to_csv("student_data.csv", index=False)
    print("DataFrame exported as 'student_data.csv' successfully!")


# ----------------------- MAIN ------------------------------
def main():
    df = create_dataframe()
    view_dataframe_info(df)
    access_rows_columns(df)
    add_new_column(df)
    modify_values(df)
    filter_and_sort(df)
    describe_stats(df)
    df = delete_column(df)
    export_to_csv(df)
    print("\n✅ Experiment Completed Successfully!")


if __name__ == "__main__":
    main()
