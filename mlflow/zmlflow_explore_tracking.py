"""
MLflow Tracking API Exploration

Demonstrates important MLflow APIs:

1. Tracking URI
2. Experiment management
3. Run management
4. Parameter logging
5. Metric logging
6. Tag logging
7. Dataset tracking
8. Artifact logging
9. Model logging
10. Active run inspection
11. Run search and comparison
12. Model Registry search by alias
13. Retrieve champion model information

Run:
    python3 mlflow/tracking_api_demo.py

MLflow UI:
    mlflow server \
        --backend-store-uri sqlite:///mlflow.db \
        --port 9001

Open:
    http://127.0.0.1:9001
"""

import os
import json

import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

from mlflow import MlflowClient

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# ============================================================
# 1. MLFLOW TRACKING CONFIGURATION
# ============================================================

TRACKING_URI = "sqlite:///mlflow.db"

EXPERIMENT_NAME = "Insurance Approval Tracking API"

REGISTERED_MODEL_NAME = "insurance-approval-model"

CHAMPION_ALIAS = "champion"


mlflow.set_tracking_uri(TRACKING_URI)

mlflow.set_experiment(EXPERIMENT_NAME)


print("======================================")
print("MLflow Configuration")
print("======================================")

print(
    f"Tracking URI : "
    f"{mlflow.get_tracking_uri()}"
)

print(
    f"Experiment   : "
    f"{EXPERIMENT_NAME}"
)

print(
    f"Registered Model : "
    f"{REGISTERED_MODEL_NAME}"
)

print(
    f"Champion Alias   : "
    f"{CHAMPION_ALIAS}"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

train = pd.read_csv(
    "data/train.csv"
)

test = pd.read_csv(
    "data/test.csv"
)


TARGET = "approved"


X_train = train.drop(
    columns=[TARGET]
)

y_train = train[TARGET]


X_test = test.drop(
    columns=[TARGET]
)

y_test = test[TARGET]


# ============================================================
# 3. CREATE MLFLOW DATASET OBJECTS
# ============================================================

train_dataset = mlflow.data.from_pandas(
    train,
    source="data/train.csv",
    name="insurance-training-data",
    targets=TARGET,
)


test_dataset = mlflow.data.from_pandas(
    test,
    source="data/test.csv",
    name="insurance-test-data",
    targets=TARGET,
)


# ============================================================
# 4. MODEL CONFIGURATION
# ============================================================

MODELS = {

    "DecisionTree": (
        DecisionTreeClassifier,
        {
            "max_depth": 6
        },
    ),

    "RandomForest": (
        RandomForestClassifier,
        {
            "n_estimators": 200,
            "max_depth": 8,
        },
    ),

    "GradientBoosting": (
        GradientBoostingClassifier,
        {
            "n_estimators": 150,
            "learning_rate": 0.08,
            "max_depth": 3,
        },
    ),

}


# ============================================================
# 5. TRAIN AND TRACK EACH MODEL
# ============================================================

best_acc = -1

best_name = None

best_run_id = None


for name, (ModelClass, params) in MODELS.items():

    print("\n======================================")
    print(f"Training: {name}")
    print("======================================")


    # --------------------------------------------------------
    # Start MLflow Run
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name=name
    ) as run:


        # ====================================================
        # A. RUN MANAGEMENT
        # ====================================================

        print(
            f"Run ID: "
            f"{run.info.run_id}"
        )


        # Get currently active run

        active_run = mlflow.active_run()


        if active_run is not None:

            print(
                f"Active Run ID: "
                f"{active_run.info.run_id}"
            )


        # ====================================================
        # B. LOG PARAMETERS
        # ====================================================

        # Log model name

        mlflow.log_param(
            "model_type",
            name
        )


        # Log all model hyperparameters

        mlflow.log_params(
            params
        )


        # Additional experiment parameters

        mlflow.log_params(
            {
                "random_state": 42,
                "target": TARGET,
                "train_rows": len(train),
                "test_rows": len(test),
            }
        )


        # ====================================================
        # C. LOG TAGS
        # ====================================================

        mlflow.set_tags(
            {
                "model_family": name,
                "experiment_type": "model_comparison",
                "dataset": "insurance",
                "environment": "development",
                "training_framework": "scikit-learn",
            }
        )


        # ====================================================
        # D. LOG DATASET INFORMATION
        # ====================================================

        mlflow.log_input(
            train_dataset,
            context="training",
        )


        mlflow.log_input(
            test_dataset,
            context="testing",
        )


        # ====================================================
        # E. TRAIN MODEL
        # ====================================================

        model = ModelClass(
            **params,
            random_state=42,
        )


        model.fit(
            X_train,
            y_train,
        )


        # ====================================================
        # F. PREDICTION
        # ====================================================

        predictions = model.predict(
            X_test
        )


        probabilities = (
            model.predict_proba(X_test)[:, 1]
        )


        # ====================================================
        # G. CALCULATE METRICS
        # ====================================================

        accuracy = accuracy_score(
            y_test,
            predictions,
        )


        precision = precision_score(
            y_test,
            predictions,
            zero_division=0,
        )


        recall = recall_score(
            y_test,
            predictions,
            zero_division=0,
        )


        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0,
        )


        auc = roc_auc_score(
            y_test,
            probabilities,
        )


        # ====================================================
        # H. LOG METRICS
        # ====================================================

        # Log accuracy individually

        mlflow.log_metric(
            "accuracy",
            accuracy,
        )


        # Log multiple metrics together

        mlflow.log_metrics(
            {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": auc,
            }
        )


        # ====================================================
        # I. CREATE ARTIFACT DIRECTORY
        # ====================================================

        artifact_dir = ".mlflow_artifacts"


        os.makedirs(
            artifact_dir,
            exist_ok=True,
        )


        # ====================================================
        # J. CONFUSION MATRIX ARTIFACT
        # ====================================================

        cm = confusion_matrix(
            y_test,
            predictions,
        )


        plt.figure()


        plt.imshow(cm)


        plt.title(
            f"Confusion Matrix - {name}"
        )


        plt.xlabel(
            "Predicted"
        )


        plt.ylabel(
            "Actual"
        )


        plt.colorbar()


        plt.xticks(
            [0, 1],
            ["Rejected", "Approved"],
        )


        plt.yticks(
            [0, 1],
            ["Rejected", "Approved"],
        )


        for i in range(2):

            for j in range(2):

                plt.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                )


        plt.tight_layout()


        confusion_matrix_path = os.path.join(
            artifact_dir,
            f"{name}_confusion_matrix.png",
        )


        plt.savefig(
            confusion_matrix_path
        )


        plt.close()


        # ====================================================
        # K. LOG SINGLE ARTIFACT
        # ====================================================

        mlflow.log_artifact(
            confusion_matrix_path,
            artifact_path="evaluation",
        )


        # ====================================================
        # L. FEATURE IMPORTANCE ARTIFACT
        # ====================================================

        if hasattr(
            model,
            "feature_importances_"
        ):

            feature_importance = pd.DataFrame(
                {
                    "feature": X_train.columns,
                    "importance":
                        model.feature_importances_,
                }
            )


            feature_importance = (
                feature_importance
                .sort_values(
                    "importance",
                    ascending=False,
                )
            )


            feature_importance_path = os.path.join(
                artifact_dir,
                f"{name}_feature_importance.csv",
            )


            feature_importance.to_csv(
                feature_importance_path,
                index=False,
            )


            mlflow.log_artifact(
                feature_importance_path,
                artifact_path="feature_importance",
            )


        # ====================================================
        # M. SAVE EVALUATION SUMMARY
        # ====================================================

        evaluation = {

            "model": name,

            "accuracy": accuracy,

            "precision": precision,

            "recall": recall,

            "f1_score": f1,

            "roc_auc": auc,

        }


        evaluation_path = os.path.join(
            artifact_dir,
            f"{name}_evaluation.json",
        )


        with open(
            evaluation_path,
            "w",
        ) as f:

            json.dump(
                evaluation,
                f,
                indent=4,
            )


        mlflow.log_artifact(
            evaluation_path,
            artifact_path="evaluation",
        )


        # ====================================================
        # N. LOG MODEL
        # ====================================================

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=X_train.head(5),
            serialization_format="skops",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ],
        )


        print(
            f"Model URI: "
            f"{model_info.model_uri}"
        )


        # ====================================================
        # O. ARTIFACT URI
        # ====================================================

        artifact_uri = mlflow.get_artifact_uri()


        print(
            f"Artifact URI: "
            f"{artifact_uri}"
        )


        # ====================================================
        # P. PRINT RESULTS
        # ====================================================

        print(
            f"{name}: "
            f"accuracy={accuracy:.4f}, "
            f"precision={precision:.4f}, "
            f"recall={recall:.4f}, "
            f"f1={f1:.4f}, "
            f"auc={auc:.4f}"
        )


        # ====================================================
        # Q. SELECT BEST MODEL
        # ====================================================

        if accuracy > best_acc:

            best_acc = accuracy

            best_name = name

            best_run_id = (
                run.info.run_id
            )


# ============================================================
# 6. LAST ACTIVE RUN
# ============================================================

last_run = mlflow.last_active_run()


print("\n======================================")
print("Last Active Run")
print("======================================")


if last_run is not None:

    print(
        f"Run ID: "
        f"{last_run.info.run_id}"
    )


    print(
        f"Status: "
        f"{last_run.info.status}"
    )


# ============================================================
# 7. BEST MODEL
# ============================================================

print("\n======================================")
print("BEST MODEL")
print("======================================")


print(
    f"Model     : "
    f"{best_name}"
)


print(
    f"Run ID    : "
    f"{best_run_id}"
)


print(
    f"Accuracy  : "
    f"{best_acc:.4f}"
)


# ============================================================
# 8. SEARCH MLFLOW RUNS
# ============================================================

print("\n======================================")
print("SEARCHING MLFLOW RUNS")
print("======================================")


runs = mlflow.search_runs(

    experiment_names=[
        EXPERIMENT_NAME
    ],

    order_by=[
        "metrics.accuracy DESC"
    ],

)


print(
    runs[
        [
            "run_id",
            "tags.mlflow.runName",
            "params.model_type",
            "metrics.accuracy",
            "metrics.precision",
            "metrics.recall",
            "metrics.f1_score",
            "metrics.roc_auc",
        ]
    ].head(10)
)


# ============================================================
# 9. SEARCH ONLY HIGH-PERFORMING RUNS
# ============================================================

print("\n======================================")
print("RUNS WITH ACCURACY >= 0.80")
print("======================================")


high_accuracy_runs = mlflow.search_runs(

    experiment_names=[
        EXPERIMENT_NAME
    ],

    filter_string=(
        "metrics.accuracy >= 0.80"
    ),

    order_by=[
        "metrics.accuracy DESC"
    ],

)


if len(high_accuracy_runs) > 0:

    print(
        high_accuracy_runs[
            [
                "run_id",
                "params.model_type",
                "metrics.accuracy",
                "metrics.f1_score",
            ]
        ]
    )


else:

    print(
        "No runs satisfy the condition."
    )


# ============================================================
# 10. RETRIEVE BEST RUN
# ============================================================

best_run = mlflow.get_run(
    best_run_id
)


print("\n======================================")
print("BEST RUN DETAILS")
print("======================================")


print(
    f"Run ID: "
    f"{best_run.info.run_id}"
)


print(
    f"Status: "
    f"{best_run.info.status}"
)


print(
    f"Parameters: "
    f"{best_run.data.params}"
)


print(
    f"Metrics: "
    f"{best_run.data.metrics}"
)


print(
    f"Tags: "
    f"{best_run.data.tags}"
)


# ============================================================
# 11. SEARCH MODEL REGISTRY BY ALIAS
# ============================================================

print("\n======================================")
print("SEARCH MODEL REGISTRY")
print("======================================")


client = MlflowClient()


print(
    f"Registered Model : "
    f"{REGISTERED_MODEL_NAME}"
)


print(
    f"Alias            : "
    f"{CHAMPION_ALIAS}"
)


try:

    champion_version = (
        client.get_model_version_by_alias(
            name=REGISTERED_MODEL_NAME,
            alias=CHAMPION_ALIAS,
        )
    )


    print("\nChampion Model Found")
    print("--------------------------------------")


    print(
        f"Model Name : "
        f"{champion_version.name}"
    )


    print(
        f"Version    : "
        f"{champion_version.version}"
    )


    print(
        f"Alias      : "
        f"{CHAMPION_ALIAS}"
    )


    print(
        f"Run ID     : "
        f"{champion_version.run_id}"
    )


    print(
        f"Source     : "
        f"{champion_version.source}"
    )


    print(
        f"Model URI  : "
        f"models:/{REGISTERED_MODEL_NAME}"
        f"@{CHAMPION_ALIAS}"
    )


except Exception as e:

    champion_version = None


    print(
        "\nNo champion model found."
    )


    print(
        f"Reason: {e}"
    )


# ============================================================
# 12. LOAD CHAMPION MODEL
# ============================================================

if champion_version is not None:

    print("\n======================================")
    print("LOAD CHAMPION MODEL")
    print("======================================")


    champion_model_uri = (
        f"models:/{REGISTERED_MODEL_NAME}"
        f"@{CHAMPION_ALIAS}"
    )


    print(
        f"Loading: "
        f"{champion_model_uri}"
    )


    champion_model = (
        mlflow.sklearn.load_model(
            champion_model_uri
        )
    )


    print(
        "Champion model loaded successfully."
    )


    # --------------------------------------------------------
    # Test the champion model
    # --------------------------------------------------------

    champion_predictions = (
        champion_model.predict(X_test)
    )


    champion_probabilities = (
        champion_model.predict_proba(
            X_test
        )[:, 1]
    )


    champion_accuracy = (
        accuracy_score(
            y_test,
            champion_predictions,
        )
    )


    champion_auc = (
        roc_auc_score(
            y_test,
            champion_probabilities,
        )
    )


    print(
        f"Champion Accuracy : "
        f"{champion_accuracy:.4f}"
    )


    print(
        f"Champion ROC-AUC  : "
        f"{champion_auc:.4f}"
    )


# ============================================================
# 13. COMPLETED
# ============================================================

print("\n======================================")
print("TRACKING COMPLETED")
print("======================================")