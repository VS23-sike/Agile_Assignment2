
import os

import mlflow
import mlflow.sklearn

from app.model import create_model


def main():
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000"
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Student-Performance-Prediction")

    with mlflow.start_run() as run:
        model, metrics = create_model()

        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("scaler", "StandardScaler")
        mlflow.log_param("test_size", 0.25)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("mean_absolute_error", metrics["mae"])
        mlflow.log_metric("r2_score", metrics["r2"])

        mlflow.sklearn.log_model(
            model,
            artifact_path="student-performance-model"
        )

        print("MLflow experiment completed successfully!")
        print(f"Tracking URI: {tracking_uri}")
        print(f"MAE: {metrics['mae']:.2f}")
        print(f"R2 Score: {metrics['r2']:.3f}")
        print(f"Run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
