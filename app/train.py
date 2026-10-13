import os

import mlflow
import mlflow.sklearn

from app.model import create_model


def main():
    # Connect to the MLflow tracking server
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000",
    )

    mlflow.set_tracking_uri(tracking_uri)

    # Set the experiment
    experiment_name = "Student-Performance-Prediction"
    mlflow.set_experiment(experiment_name)

    # Train, track, and register the model
    with mlflow.start_run() as run:
        model, metrics = create_model()

        # Log model parameters
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("scaler", "StandardScaler")
        mlflow.log_param("test_size", 0.25)
        mlflow.log_param("random_state", 42)

        # Log evaluation metrics
        mlflow.log_metric(
            "mean_absolute_error",
            metrics["mae"],
        )
        mlflow.log_metric(
            "r2_score",
            metrics["r2"],
        )

        # Log and register the trained model
        mlflow.sklearn.log_model(
            model,
            artifact_path="student-performance-model",
            registered_model_name="StudentPerformanceModel",
        )

        # Display results
        print("\nMLflow experiment completed successfully!")
        print(f"Tracking URI: {tracking_uri}")
        print(f"Experiment: {experiment_name}")
        print(f"MAE: {metrics['mae']:.2f}")
        print(f"R2 Score: {metrics['r2']:.3f}")
        print(f"Run ID: {run.info.run_id}")
        print("Registered model: StudentPerformanceModel")


if __name__ == "__main__":
    main()
