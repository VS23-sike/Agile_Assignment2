from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, make_asgi_app

from app.model import create_model

app = FastAPI(
    title="Student Performance Prediction API",
    description="Predicts an example student exam score using study hours, attendance, and previous score.",
    version="1.0.0",
)

# Train the model when the application starts.
model, model_metrics = create_model()

# Prometheus metrics
PREDICTION_REQUESTS = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time spent processing prediction requests",
)

# Expose metrics at /metrics for Prometheus.
app.mount("/metrics", make_asgi_app())


class StudentInput(BaseModel):
    study_hours: float = Field(ge=0, le=24)
    attendance: float = Field(ge=0, le=100)
    previous_score: float = Field(ge=0, le=100)


@app.get("/")
def home():
    return {
        "message": "Student Performance Prediction API is running",
        "docs": "/docs",
        "metrics": "/metrics",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/model-info")
def model_info():
    return {
        "model": "StandardScaler + LinearRegression",
        "evaluation_metrics": model_metrics,
    }


@app.post("/predict")
def predict(student: StudentInput):
    PREDICTION_REQUESTS.inc()

    try:
        with PREDICTION_LATENCY.time():
            features = [[
                student.study_hours,
                student.attendance,
                student.previous_score,
            ]]
            prediction = float(model.predict(features)[0])

        # Keep the illustrative prediction within the expected score range.
        prediction = max(0.0, min(100.0, prediction))

        return {
            "predicted_score": round(prediction, 2),
            "note": "This is an illustrative model, not a validated student assessment.",
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed",
        ) from exc