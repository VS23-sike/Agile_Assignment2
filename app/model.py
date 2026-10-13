import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_model():
    """Create and train a student performance prediction model."""

    # Example training data:
    # [hours studied, attendance percentage, previous exam score]
    X = np.array([
        [1, 55, 40],
        [2, 60, 45],
        [3, 65, 50],
        [4, 70, 55],
        [5, 75, 60],
        [6, 80, 65],
        [7, 85, 70],
        [8, 90, 75],
        [9, 95, 85],
        [2, 50, 35],
        [4, 80, 65],
        [6, 70, 55],
        [7, 90, 80],
        [3, 75, 60],
        [8, 85, 78],
        [5, 65, 52],
        [9, 98, 90],
        [1, 45, 30],
        [6, 88, 72],
        [4, 60, 48],
    ], dtype=float)

    # Target: example final exam scores out of 100
    y = np.array([
        42, 48, 53, 58, 65,
        70, 76, 83, 92, 38,
        64, 59, 84, 61, 86,
        55, 95, 32, 78, 49
    ], dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }

    return model, metrics


if __name__ == "__main__":
    trained_model, evaluation = create_model()
    print("Student Performance Model trained successfully!")
    print(f"Mean Absolute Error: {evaluation['mae']:.2f}")
    print(f"R² Score: {evaluation['r2']:.3f}")