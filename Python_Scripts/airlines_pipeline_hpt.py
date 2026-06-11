import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import optuna
# pip install optuna-integration[mlflow]
from optuna.integration.mlflow import MLflowCallback

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import joblib
import time

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import warnings
warnings.filterwarnings("ignore")

# ── Load Dataset ──────────────────────────────────────────────────────────────
data = pd.read_csv("datasets/airlines_top10_features.csv")
# Data Cleaning
data = data.drop_duplicates()

# Segregate features and target
X = data.drop("Satisfaction", axis=1)
y = data["Satisfaction"]

# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Identify column types
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols   = X.select_dtypes(include=['number']).columns.tolist()

print("Categorical columns:", categorical_cols)
print("Numerical columns:", numerical_cols)


# ── Preprocessor factory ──────────────────────────────────────────────────────
def make_preprocessor(scaler):
    return ColumnTransformer(transformers=[
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols),
        ('num', scaler, numerical_cols)
    ])


# ── Pipeline ──────────────────────────────────────────────────────────────────
pipeline = Pipeline([
    ('Preprocessor', make_preprocessor(StandardScaler())),
    ('Model', KNeighborsClassifier())
])


# ── Objective functions ───────────────────────────────────────────────────────

def objective_knn(trial):
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
    scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
    pipeline.set_params(
        Preprocessor=make_preprocessor(scaler),
        Model=KNeighborsClassifier(),
        Model__n_neighbors=trial.suggest_int('n_neighbors', 3, 21, 2),
        Model__weights=trial.suggest_categorical('weights', ['uniform', 'distance']),
        Model__p=trial.suggest_int('p', 1, 3)
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    return cross_val_score(pipeline, X_train, y_train, scoring='accuracy', cv=skf).mean()


def objective_dt(trial):
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
    scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
    pipeline.set_params(
        Preprocessor=make_preprocessor(scaler),
        Model=DecisionTreeClassifier(),
        Model__criterion=trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
        Model__max_depth=trial.suggest_int('max_depth', 2, 30),
        Model__min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
        Model__min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
        Model__max_features=trial.suggest_categorical('max_features', [None, 'sqrt', 'log2'])
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    return cross_val_score(pipeline, X_train, y_train, scoring='accuracy', cv=skf).mean()


def objective_svm(trial):
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
    scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid'])
    params = {
        'C': trial.suggest_float('C', 1e-3, 1e2, log=True),
        'kernel': kernel
    }
    if kernel in ['rbf', 'poly', 'sigmoid']:
        params['gamma'] = trial.suggest_float('gamma', 1e-4, 1e-1, log=True)
    if kernel == 'poly':
        params['degree'] = trial.suggest_int('degree', 2, 5)
    pipeline.set_params(
        Preprocessor=make_preprocessor(scaler),
        Model=SVC(**params)
    )
    skf = StratifiedKFold(n_splits=2, shuffle=True)
    return cross_val_score(pipeline, X_train, y_train, scoring='accuracy', cv=skf).mean()


def objective_gnb(trial):
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
    scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
    pipeline.set_params(
        Preprocessor=make_preprocessor(scaler),
        Model=GaussianNB(
            var_smoothing=trial.suggest_float('var_smoothing', 1e-11, 1e-7, log=True)
        )
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    return cross_val_score(pipeline, X_train, y_train, scoring='accuracy', cv=skf).mean()


def objective_rf(trial):
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
    scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
    pipeline.set_params(
        Preprocessor=make_preprocessor(scaler),
        Model=RandomForestClassifier(
            n_estimators=trial.suggest_int('n_estimators', 100, 500, step=50),
            criterion=trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
            max_depth=trial.suggest_int('max_depth', 5, 40),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
            max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            bootstrap=trial.suggest_categorical('bootstrap', [True, False]),
            random_state=42,
            n_jobs=-1
        )
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    return cross_val_score(pipeline, X_train, y_train, scoring='accuracy', cv=skf).mean()


def objective_gb(trial):
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
    scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
    pipeline.set_params(
        Preprocessor=make_preprocessor(scaler),
        Model=GradientBoostingClassifier(
            n_estimators=trial.suggest_int('n_estimators', 100, 500, step=50),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            max_depth=trial.suggest_int('max_depth', 2, 10),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
            max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            random_state=42
        )
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    return cross_val_score(pipeline, X_train, y_train, scoring='accuracy', cv=skf).mean()


# ── Model registry ────────────────────────────────────────────────────────────
objectives = {
    "KNN":             objective_knn,
    "DecisionTree":    objective_dt,
    "SVM":             objective_svm,
    "GaussianNB":      objective_gnb,
    "RandomForest":    objective_rf,
    "GradientBoosting": objective_gb
}

model_dict  = {name: i for i, name in enumerate(objectives.keys())}
scaler_dict = {"standard": 0, "minmax": 1}

# ── MLflow experiment ─────────────────────────────────────────────────────────
mlflow.set_experiment("Airlines_PL_RUNS")

results = {}

# ── Main loop ─────────────────────────────────────────────────────────────────
for model_name, obj_fn in objectives.items():
    print(f"\n--- Optimizing {model_name} ---")

    mlflow_cb = MLflowCallback(
        tracking_uri=None,
        metric_name="cv_accuracy",
        mlflow_kwargs={"nested": True}
    )

    study = optuna.create_study(direction="maximize")

    start_fit = time.time()
    study.optimize(obj_fn, n_trials=20, callbacks=[mlflow_cb])
    fit_time = time.time() - start_fit

    print(f"Best CV accuracy for {model_name}: {study.best_value:.4f}")
    best_params = study.best_params
    results[model_name] = {"best_params": best_params, "best_cv_accuracy": study.best_value}

    # ── Re-configure pipeline with best params ────────────────────────────────
    scaler = StandardScaler() if best_params["scaler_type"] == "standard" else MinMaxScaler()

    if model_name == "KNN":
        pipeline.set_params(
            Preprocessor=make_preprocessor(scaler),
            Model=KNeighborsClassifier(),
            Model__n_neighbors=best_params["n_neighbors"],
            Model__weights=best_params["weights"],
            Model__p=best_params["p"]
        )
    elif model_name == "DecisionTree":
        pipeline.set_params(
            Preprocessor=make_preprocessor(scaler),
            Model=DecisionTreeClassifier(),
            Model__criterion=best_params["criterion"],
            Model__max_depth=best_params["max_depth"],
            Model__min_samples_split=best_params["min_samples_split"],
            Model__min_samples_leaf=best_params["min_samples_leaf"],
            Model__max_features=best_params["max_features"]
        )
    elif model_name == "SVM":
        svm_params = {"kernel": best_params["kernel"], "C": best_params["C"]}
        if best_params["kernel"] in ["rbf", "poly", "sigmoid"]:
            svm_params["gamma"] = best_params["gamma"]
        if best_params["kernel"] == "poly":
            svm_params["degree"] = best_params["degree"]
        pipeline.set_params(
            Preprocessor=make_preprocessor(scaler),
            Model=SVC(**svm_params)
        )
    elif model_name == "GaussianNB":
        pipeline.set_params(
            Preprocessor=make_preprocessor(scaler),
            Model=GaussianNB(),
            Model__var_smoothing=best_params["var_smoothing"]
        )
    elif model_name == "RandomForest":
        pipeline.set_params(
            Preprocessor=make_preprocessor(scaler),
            Model=RandomForestClassifier(random_state=42, n_jobs=-1),
            Model__n_estimators=best_params["n_estimators"],
            Model__criterion=best_params["criterion"],
            Model__max_depth=best_params["max_depth"],
            Model__min_samples_split=best_params["min_samples_split"],
            Model__min_samples_leaf=best_params["min_samples_leaf"],
            Model__max_features=best_params["max_features"],
            Model__bootstrap=best_params["bootstrap"]
        )
    elif model_name == "GradientBoosting":
        pipeline.set_params(
            Preprocessor=make_preprocessor(scaler),
            Model=GradientBoostingClassifier(random_state=42),
            Model__n_estimators=best_params["n_estimators"],
            Model__learning_rate=best_params["learning_rate"],
            Model__max_depth=best_params["max_depth"],
            Model__min_samples_split=best_params["min_samples_split"],
            Model__min_samples_leaf=best_params["min_samples_leaf"],
            Model__max_features=best_params["max_features"],
            Model__subsample=best_params["subsample"]
        )

    # ── Train final model ─────────────────────────────────────────────────────
    pipeline.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    start_test = time.time()
    y_pred = pipeline.predict(X_test)
    test_time = time.time() - start_test

    train_acc = pipeline.score(X_train, y_train)
    test_acc  = accuracy_score(y_test, y_pred)

    print(f"{model_name} Training Accuracy: {train_acc:.4f}, Testing Accuracy: {test_acc:.4f}")
    print(f"{model_name} Fit Time: {fit_time:.2f}s, Test Time: {test_time:.4f}s")

    # ── Save model to get size ────────────────────────────────────────────────
    model_path = f"{model_name}_final_model.pkl"
    joblib.dump(pipeline, model_path)
    model_size = os.path.getsize(model_path)

    # ── Log to MLflow (parent run per model) ──────────────────────────────────
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("scaler_type", best_params["scaler_type"])
        mlflow.log_params({k: v for k, v in best_params.items() if k != "scaler_type"})

        mlflow.log_metric("model_id",        model_dict[model_name])
        mlflow.log_metric("scaler_id",       scaler_dict[best_params["scaler_type"]])
        mlflow.log_metric("best_cv_accuracy", study.best_value)
        mlflow.log_metric("train_accuracy",  train_acc)
        mlflow.log_metric("test_accuracy",   test_acc)
        mlflow.log_metric("train_time",      fit_time)
        mlflow.log_metric("test_time",       test_time)
        mlflow.log_metric("model_size",      model_size)

        mlflow.sklearn.log_model(pipeline, name=f"{model_name}_airlines_model")

    os.remove(model_path)

    results[model_name].update({
        "train_accuracy":   train_acc,
        "test_accuracy":    test_acc,
        "fit_time":         fit_time,
        "test_time":        test_time,
        "model_size_bytes": model_size
    })
