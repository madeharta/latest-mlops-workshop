from pathlib import Path
from datetime import datetime, timezone
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.metrics import ValueDrift
from evidently.presets import DataDriftPreset, ClassificationPreset
from evidently.ui.workspace import Workspace

BASE_DIR = Path(__file__).resolve().parent.parent

# Data
REFERENCE_DATA = BASE_DIR / "data" / "test.csv"
CURRENT_DATA = BASE_DIR / "data" / "production.csv"

# Local model
LOCAL_MODEL = BASE_DIR / "api" / "models" / "model.pkl"

# Evidently local workspace
WORKSPACE_PATH = BASE_DIR / "monitoring" / ".workspace"

# HTML reports
REPORT_DIR = BASE_DIR / "monitoring" / ".reports"

# Evidently project
PROJECT_NAME = "Insurance Approval Monitoring"

# MLflow
MLFLOW_TRACKING_URI = "http://127.0.0.1:9001"
MLFLOW_MODEL_URI = "models:/insurance-approval-model@champion"

FEATURES = ["age","income","bmi","tenure_months","num_claims","credit_score","policy_value","risk_score",]

# Retraining is triggered when at least 30% of features drift
DATA_DRIFT_THRESHOLD = 0.30

# Retraining is triggered when accuracy drops by >= 3 percentage points
ACCURACY_DROP_THRESHOLD = 0.03

# Simple relative-change threshold used by the
# teaching/retraining policy below
FEATURE_DRIFT_THRESHOLD = 0.10

def load_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model(MLFLOW_MODEL_URI)
    print("Champion model loaded successfully.")
    return model

def load_data():
    reference = pd.read_csv(REFERENCE_DATA)
    current = pd.read_csv(CURRENT_DATA)

    return reference, current

def validate_data(df, dataset_name):
    required_columns = (FEATURES+ ["approved"])

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"{dataset_name} is missing "f"required columns: "f"{missing_columns}")

    if df.empty:
        raise ValueError(f"{dataset_name} is empty.")

def generate_predictions(model,df):
    X = df[FEATURES]
    prediction = model.predict(X)
    prediction_proba = (model.predict_proba(X)[:, 1])
    result = df.copy()
    result["prediction"] = prediction
    result["prediction_proba"] = (prediction_proba)

    return result

def create_data_definition():
    data_definition = DataDefinition(
        numerical_columns=FEATURES,
        classification=[
            BinaryClassification(
                target="approved",
                prediction_labels="prediction",
                prediction_probas=("prediction_proba"),
                pos_label=1,
            )
        ],
    )

    return data_definition

def create_evidently_datasets(reference, current):
    data_definition = (create_data_definition())
    reference_dataset = (
        Dataset.from_pandas(
            reference,
            data_definition=data_definition,
        )
    )

    current_dataset = (
        Dataset.from_pandas(
            current,
            data_definition=data_definition,
        )
    )

    return (reference_dataset,current_dataset)

def run_monitoring(reference_dataset, current_dataset):
    report = Report(
        metrics=[
            DataDriftPreset(columns=FEATURES),
            ValueDrift(column="prediction"),
            ClassificationPreset(),
        ]
    )
    snapshot = report.run(current_dataset,reference_dataset)

    return snapshot

def save_html_report(snapshot):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = (REPORT_DIR/ f"monitoring_{timestamp}.html")

    snapshot.save_html(str(report_path))
    print()
    print("HTML report saved:")
    print(report_path)

    return report_path

def get_workspace():
    WORKSPACE_PATH.mkdir(parents=True, exist_ok=True)
    workspace = Workspace.create(str(WORKSPACE_PATH))

    return workspace

def get_project(workspace):
    projects = (workspace.search_project(PROJECT_NAME))

    if projects:
        project = projects[0]
        print(f"Using existing Evidently "f"project: {PROJECT_NAME}")
    else:
        project = (
            workspace.create_project(
                PROJECT_NAME,
                description=("Monitoring ML model using evidently."),
            )
        )
        print(f"Created Evidently project: "f"{PROJECT_NAME}")

    return project

def save_to_workspace(snapshot):
    workspace = get_workspace()
    project = get_project(workspace)
    workspace.add_run(project.id,snapshot)

def calculate_retraining_signal(reference,current):
    # data drift >= 30% OR accuracy drop >= 3 percentage points
    reference_correct = (reference["approved"] == reference["prediction"])
    reference_accuracy = (reference_correct.mean())
    current_correct = (current["approved"]== current["prediction"])
    current_accuracy = (current_correct.mean())
    accuracy_drop = (reference_accuracy- current_accuracy)

    # --------------------------------------------------------
    # SIMPLE FEATURE DRIFT
    # --------------------------------------------------------
    drifted_features = 0
    for feature in FEATURES:
        reference_mean = (reference[feature].mean())
        current_mean = (current[feature].mean())

        # Avoid division by zero
        if reference_mean == 0:
            continue

        relative_change = (abs(current_mean - reference_mean) / abs(reference_mean))

        if (relative_change >= FEATURE_DRIFT_THRESHOLD):
            drifted_features += 1

    # --------------------------------------------------------
    # DATA DRIFT SHARE
    # --------------------------------------------------------

    data_drift_share = (drifted_features / len(FEATURES))

    # --------------------------------------------------------
    # RETRAINING DECISION
    # --------------------------------------------------------

    retraining_required = bool((data_drift_share >= DATA_DRIFT_THRESHOLD)
        or (accuracy_drop >= ACCURACY_DROP_THRESHOLD))

    # --------------------------------------------------------
    # RETURN ONLY JSON-SAFE TYPES
    # --------------------------------------------------------

    return {
        "reference_accuracy": float(reference_accuracy),
        "current_accuracy": float(current_accuracy),
        "accuracy_drop": float(accuracy_drop),
        "drifted_features": int(drifted_features),
        "data_drift_share": float(data_drift_share),
        "retraining_required": bool(retraining_required),
    }

def print_summary(results):
    print()
    print("=" * 60)
    print("MODEL MONITORING SUMMARY")
    print("=" * 60)

    print(f"Reference accuracy : "f"{results['reference_accuracy']:.4f}")
    print(f"Current accuracy   : "f"{results['current_accuracy']:.4f}")
    print(f"Accuracy drop      : "f"{results['accuracy_drop']:.4f}")
    print(f"Drifted features   : "f"{results['drifted_features']} "f"/ {len(FEATURES)}")
    print(f"Data drift share   : "f"{results['data_drift_share']:.2%}")
    print(f"Retraining needed  : "f"{results['retraining_required']}")
    print("=" * 60)

def save_json_summary(results):
    REPORT_DIR.mkdir(parents=True, exist_ok=True,)
    summary_path = (REPORT_DIR / "latest_monitoring_summary.json")

    with open(summary_path,"w",encoding="utf-8") as f:
        json.dump(results,f,indent=4)

    print()
    print(f"Summary saved to: "f"{summary_path}")
    return summary_path

def main():
    model = load_model()
    reference, current = (load_data())
    validate_data(reference,"Reference dataset")
    validate_data(current,"Current dataset")
    reference = (generate_predictions(model,reference))
    current = (generate_predictions(model,current))

    # --------------------------------------------------------
    # Run evidently monitoringg
    # --------------------------------------------------------

    (reference_dataset,current_dataset) = create_evidently_datasets(reference,current)
    snapshot = run_monitoring(reference_dataset,current_dataset)
    save_html_report(snapshot)
    save_to_workspace(snapshot)
    results = (calculate_retraining_signal(reference,current))

    print_summary(results)
    save_json_summary(results)

if __name__ == "__main__":
    main()