# ==========================================
# VEGETABLE PRICE PREDICTION PROJECT
# ==========================================

# IMPORT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# ==========================================
# COLUMN NAMES
# ==========================================

columns = [
    "Vegetable_Name",
    "Category",
    "Local_Type",
    "Other1",
    "Other2",
    "Date",
    "Unit",
    "Max_Price",
    "Min_Price",
    "Avg_Price"
]


# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(
    "tarkari.csv",
    names=columns
)


# ==========================================
# REMOVE BROKEN FIRST ROW
# ==========================================

df = df.iloc[1:]


# ==========================================
# SHOW FIRST 5 ROWS
# ==========================================

print("\nFIRST 5 ROWS:\n")
print(df.head())


# ==========================================
# CLEAN VEGETABLE NAMES
# ==========================================

df["Vegetable_Name"] = (
    df["Vegetable_Name"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# ==========================================
# CONVERT DATE
# ==========================================

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

df = df.dropna(subset=["Date"])


# ==========================================
# CONVERT PRICE COLUMNS
# ==========================================

df["Avg_Price"] = pd.to_numeric(
    df["Avg_Price"],
    errors="coerce"
)

df["Min_Price"] = pd.to_numeric(
    df["Min_Price"],
    errors="coerce"
)

df["Max_Price"] = pd.to_numeric(
    df["Max_Price"],
    errors="coerce"
)


# ==========================================
# REMOVE INVALID PRICE ROWS
# ==========================================

df = df.dropna(subset=["Avg_Price"])


# ==========================================
# CREATE DATE FEATURES
# ==========================================

df["Day"] = df["Date"].dt.day
df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year


# ==========================================
# SHOW AVAILABLE VEGETABLES
# ==========================================

print("\nAVAILABLE VEGETABLES:\n")
print(df["Vegetable_Name"].unique())


# ==========================================
# SELECT VEGETABLE
# ==========================================
# CHANGE THIS NAME ONLY

selected_vegetable = "lemon"


# ==========================================
# FILTER DATA
# ==========================================

selected_data = df[
    df["Vegetable_Name"] == selected_vegetable
]


# ==========================================
# SHOW DATA SHAPE
# ==========================================

print("\nSELECTED DATA SHAPE:\n")
print(selected_data.shape)


# ==========================================
# SHOW SELECTED DATA
# ==========================================

print("\nSELECTED DATA:\n")
print(selected_data.head())


# ==========================================
# CHECK IF DATA EXISTS
# ==========================================

if len(selected_data) == 0:

    print("\nERROR:")
    print("No data found.")
    print("Choose another vegetable.")

else:

    # ==========================================
    # INPUT FEATURES
    # ==========================================

    X = selected_data[
        ["Day", "Month", "Year"]
    ]


    # ==========================================
    # TARGET OUTPUT
    # ==========================================

    y = selected_data["Avg_Price"]


    # ==========================================
    # SPLIT DATA
    # ==========================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


    # ==========================================
    # CREATE MODEL
    # ==========================================

    model = LinearRegression()


    # ==========================================
    # TRAIN MODEL
    # ==========================================

    model.fit(X_train, y_train)

    print("\nMODEL TRAINED SUCCESSFULLY")


    # ==========================================
    # FUTURE PREDICTION
    # ==========================================

    future_prediction = model.predict(
        [[15, 5, 2026]]
    )

    print("\nPREDICTED PRICE:\n")
    print(future_prediction[0])


    # ==========================================
    # GRAPH
    # ==========================================

    plt.figure(figsize=(12, 6))

    plt.plot(
        selected_data["Date"],
        selected_data["Avg_Price"]
    )

    plt.title(
        f"{selected_vegetable.upper()} PRICE TREND"
    )

    plt.xlabel("Date")
    plt.ylabel("Average Price")

    plt.grid(True)

    plt.show()


    print("\nPROJECT EXECUTED SUCCESSFULLY")
