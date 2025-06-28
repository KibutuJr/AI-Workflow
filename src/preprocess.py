import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def main():
    # Load raw data
    df = pd.read_csv("data/student-por.csv", sep=";")

    # Drop missing values
    df = df.dropna()

    # Scale numerical features
    numeric_cols = ["age", "absences", "G1", "G2", "G3"]
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Encode all categorical columns
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Save processed data
    df.to_csv("data/clean.csv", index=False)
    print(" Preprocessing complete. Cleaned data saved to data/clean.csv")

if __name__ == "__main__":
    main()
