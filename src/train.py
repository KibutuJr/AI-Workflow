import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def main():
    # Load cleaned data
    df = pd.read_csv("data/clean.csv")
    X = df.drop("G3", axis=1)
    y = (df["G3"] > df["G3"].median()).astype(int)  # Convert target to binary

    # 🔍 Show feature columns for API use
    feature_list = list(X.columns)
    print("\n🔢 Total Features:", len(feature_list))
    print("📋 Feature List (for /predict input):")
    print(feature_list)

    # Save feature list for deployment reference
    with open("models/features.txt", "w") as f:
        for feat in feature_list:
            f.write(feat + "\n")

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Save model
    with open("models/readmit_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("\n✅ Model trained and saved to models/readmit_model.pkl")

if __name__ == "__main__":
    main()
