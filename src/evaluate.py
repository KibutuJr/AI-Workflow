import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score

def main():
    # Load data
    df = pd.read_csv("data/clean.csv")
    X = df.drop("G3", axis=1)
    y = (df["G3"] > df["G3"].median()).astype(int)

    # Same data splits as in training
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Load model
    with open("models/readmit_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("📊 Evaluation Results:")
    print("Confusion Matrix:\n", cm)
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")

if __name__ == "__main__":
    main()
