import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import numpy as np

def log_model_to_mlflow():
    # Set experiment name
    mlflow.set_experiment("Travel_Price_Prediction")

    with mlflow.start_run():
        # Log parameters
        n_estimators = 100
        max_depth = 7
        learning_rate = 0.2
        
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("learning_rate", learning_rate)

        # Mock training data for demonstration
        X_train = np.random.rand(100, 11)
        y_train = np.random.rand(100)
        
        model = GradientBoostingRegressor(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            learning_rate=learning_rate
        )
        model.fit(X_train, y_train)

        # Mock evaluation
        y_pred = model.predict(X_train)
        mae = mean_absolute_error(y_train, y_pred)
        r2 = r2_score(y_train, y_pred)

        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Log model
        mlflow.sklearn.log_model(model, "flight_price_model")
        
        print(f"Model logged to MLflow with MAE: {mae} and R2: {r2}")

if __name__ == "__main__":
    log_model_to_mlflow()
