import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# LOAD DATA
def load_data(path):
    return pd.read_csv(path)


def train_with_tuning(df):
    # SET EXPERIMENT (WAJIB SAMA DENGAN MLFLOW PROJECT)
    mlflow.set_experiment("Widya-Experiment-Tuning")

    # TARGET
    y = (df["Quantity"] > df["Quantity"].median()).astype(int)

    # FEATURE SELECTION
    drop_cols = [
        "Invoice",
        "StockCode",
        "Description",
        "InvoiceDate",
        "CustomerID"
    ]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = X.select_dtypes(include=["int64", "float64"])

    # SPLIT DATA
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    param_grid = [
        {"n_estimators": 50, "max_depth": None},
        {"n_estimators": 100, "max_depth": None},
    ]

    # ⚠️ TIDAK ADA start_run
    for params in param_grid:
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # LOG PARAM
        mlflow.log_param("n_estimators", params["n_estimators"])
        mlflow.log_param("max_depth", params["max_depth"])

        # LOG METRIC
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

        # LOG MODEL
        mlflow.sklearn.log_model(model, "sklearn-model")

        print("Run selesai (CI):", params)


if __name__ == "__main__":
    df = load_data("processed_transaction_dataset_1000.csv")
    train_with_tuning(df)
