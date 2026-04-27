import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

from src.data_preprocessing import clean_data  # type: ignore
from src.evaluate import evaluate  # type: ignore

from sklearn.ensemble import RandomForestClassifier


def train_model(df):

    # ---------------- PREPROCESS ----------------
    df = clean_data(df)

    df["Age_Group"] = pd.cut(
        df["Victim Age"],
        bins=[0, 18, 40, 60, 100],
        labels=[0, 1, 2, 3]
    )

    df = df.drop([
        "Report Number",
        "Date Reported",
        "Date of Occurrence",
        "Time of Occurrence",
        "Crime Description",
        "Date Case Closed"
    ], axis=1)

    # ---------------- SPLIT ----------------
    le = LabelEncoder()
    y = le.fit_transform(df["Crime Domain"])
    X = df.drop("Crime Domain", axis=1)

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------- MODEL ----------------
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # ---------------- EVALUATION ----------------
    y_pred = model.predict(X_test)

    print("\n📊 Model Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

    evaluate(model, X_test, y_test)

    # ---------------- SAVE ----------------
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/trained_model.pkl", compress=3)
    joblib.dump(X.columns, "models/columns.pkl")
    joblib.dump(le, "models/label_encoder.pkl")

    print("✅ Model saved successfully!")

    return model, X_test, y_test
