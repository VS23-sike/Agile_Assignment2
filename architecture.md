
flowchart TD
    A["Agile Planning<br/>GitHub Issues"] --> B["GitHub Repository<br/>Source Code"]
    B --> C["GitHub Actions<br/>Automated CI/CD"]
    C --> D["Install Dependencies<br/>Run Pytest"]
    D --> E["Build Docker Image"]
    E --> F["Docker Compose<br/>Deployment"]

    F --> G["FastAPI<br/>Prediction API"]
    F --> H["MLflow<br/>Experiment Tracking"]
    F --> I["Prometheus<br/>Metrics Collection"]
    F --> J["Grafana<br/>Monitoring Dashboard"]

    K["Model Training<br/>app/train.py"] --> H
    H --> L["Registered Model<br/>Version Management"]
    G --> M["Prediction Endpoint<br/>POST /predict"]
    G --> I
    I --> J

    classDef planning fill:#e8def8,stroke:#7555a5,color:#24163b
    classDef pipeline fill:#dbeafe,stroke:#3973ac,color:#102b4c
    classDef deploy fill:#d9f3e5,stroke:#368568,color:#123b2c
    classDef monitor fill:#fff0d6,stroke:#b77b22,color:#573b0c

    class A planning
    class B,C,D,E pipeline
    class F,G,K,L,M deploy
    class H,I,J monitor
