
import os
import yaml
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load parameters
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_train.csv")

target = params["target"]

X = df.drop(columns=[target])
y = df[target]

# Identify categorical and numerical columns
categorical = X.select_dtypes(include=["object"]).columns.tolist()
numerical = X.select_dtypes(exclude=["object"]).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("numerical", "passthrough", numerical)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=params["test_size"],
    random_state=params["random_state"]
)

# MLflow setup
mlflow.set_experiment(params["tracking"]["experiment_name"])

# Three model configurations
base = params["model"]

configs = [
    {"name": "baseline", **base},
    {
        "name": "shallow",
        **{**base, "max_depth": 6}
    },
    {
        "name": "deeper",
        **{**base, "max_depth": 18, "n_estimators": 300}
    }
]

best_mae = float("inf")
best_model = None
best_run_id = None

for config in configs:

    model_params = {
        k: v for k, v in config.items() if k != "name"
    }

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(**model_params))
    ])

    with mlflow.start_run(run_name=config["name"]) as run:

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)

        mlflow.log_params(model_params)
        mlflow.log_metrics({
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        mlflow.sklearn.log_model(
    model,
    name="model",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)

        print("\nModel:", config["name"])
        print("MAE:", mae)
        print("RMSE:", rmse)
        print("R2:", r2)

        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_run_id = run.info.run_id

# Save best model
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model.pkl")

print("\nBEST MODEL SELECTED")
print("Best MAE:", best_mae)
print("Best Run ID:", best_run_id)

# Register the best model
with mlflow.start_run(run_name="champion-registration"):
    mlflow.log_param("selected_run_id", best_run_id)
    mlflow.log_metric("best_MAE", best_mae)

    mlflow.sklearn.log_model(
    best_model,
    name="champion_model",
    registered_model_name="wheat-yield-champion",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)
    

print("\nMODEL TRAINING AND REGISTRATION COMPLETED!")
