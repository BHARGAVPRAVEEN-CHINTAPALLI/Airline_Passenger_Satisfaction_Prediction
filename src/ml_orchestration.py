import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn import metrics
from prefect import task, flow
import os

@task
def load_data(file_path):
    """
    Load data from a CSV file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    data = pd.read_csv(file_path)
    data = data.drop_duplicates()
    return data

@task
def split_inputs_output(data, inputs, output):
    """
    Split features and target variables.
    """
    X = data[inputs]
    y = data[output]
    return X, y


@task
def split_train_test(X, y, test_size=0.3, random_state=42):
    """
    Split data into train and test sets.
    """
    return train_test_split(X, y, test_size=test_size,
                            stratify=y, random_state=random_state)


@task
def build_pipeline(X, hyperparameters):
    """
    Build preprocessing + GradientBoosting pipeline.
    Preprocessing:
      - Categorical cols → SimpleImputer (most_frequent) + OrdinalEncoder
      - Numerical cols   → SimpleImputer (median)        + MinMaxScaler
    """
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols   = X.select_dtypes(include=["number"]).columns.tolist()

    print("Categorical columns :", categorical_cols)
    print("Numerical  columns  :", numerical_cols)

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=-1))
    ])

    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  MinMaxScaler())          # best scaler from MLflow: minmax
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("cat", categorical_transformer, categorical_cols),
        ("num", numerical_transformer,   numerical_cols)
    ])

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model",        GradientBoostingClassifier(**hyperparameters))
    ])

    return pipeline


@task
def train_model(pipeline, X_train, y_train):
    """
    Fit the pipeline on training data.
    """
    pipeline.fit(X_train, y_train)
    return pipeline


@task
def evaluate_model(pipeline, X_train, y_train, X_test, y_test):
    """
    Evaluate the fitted pipeline on train and test sets.
    """
    y_train_pred = pipeline.predict(X_train)
    y_test_pred  = pipeline.predict(X_test)

    train_acc  = metrics.accuracy_score(y_train, y_train_pred)
    test_acc   = metrics.accuracy_score(y_test,  y_test_pred)
    precision  = metrics.precision_score(y_test, y_test_pred, pos_label="Satisfied")
    recall     = metrics.recall_score(y_test,    y_test_pred, pos_label="Satisfied")
    f1         = metrics.f1_score(y_test,        y_test_pred, pos_label="Satisfied")

    print("\n========== EVALUATION RESULTS ==========")
    print(f"Train Accuracy : {train_acc:.4f}")
    print(f"Test  Accuracy : {test_acc:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1 Score       : {f1:.4f}")
    print("=========================================\n")

    return train_acc, test_acc, precision, recall, f1


# ── Workflow ──────────────────────────────────────────────────────────────────
@flow(name="Airlines GradientBoosting Flow")
def workflow():

    from pathlib import Path

    DATA_PATH = Path("datasets/airlines_top10_features.csv")
    OUTPUT = "Satisfaction"

    # Best hyperparameters from MLflow (GradientBoosting)
    HYPERPARAMETERS = {
        "n_estimators"     : 450,
        "learning_rate"    : 0.28002556107273757,
        "max_depth"        : 3,
        "min_samples_split": 5,
        "min_samples_leaf" : 7,
        "max_features"     : "sqrt",
        "subsample"        : 0.9997627754830676,
        "random_state"     : 42
    }

    # Load data
    data = load_data(DATA_PATH)

    # All columns except target as inputs
    INPUTS = [col for col in data.columns if col != OUTPUT]

    # Split features and target
    X, y = split_inputs_output(data, INPUTS, OUTPUT)

    # Train / test split
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Build pipeline (preprocessor + model)
    pipeline = build_pipeline(X, HYPERPARAMETERS)

    # Train
    trained_pipeline = train_model(pipeline, X_train, y_train)

    # Evaluate
    train_acc, test_acc, precision, recall, f1 = evaluate_model(
        trained_pipeline, X_train, y_train, X_test, y_test
    )

    print("Train Accuracy :", train_acc)
    print("Test  Accuracy :", test_acc)
    print("F1 Score       :", f1)


if __name__ == "__main__":
    workflow.serve(
        name="airlines-gb-deployment",
        cron="* * * * *"
    )
