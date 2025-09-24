import pandas as pd
import numpy as np
import pickle

class WeatherForecaster:
    def __init__(self, model_path, feature_cols, data_path="agg_lag.csv"):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        
        self.agg_lag = pd.read_csv(data_path)
        self.feature_cols = feature_cols

    def forecast(self, chosen_month, lat, lon):
        # Try to get exact row
        row = self.agg_lag[
            (self.agg_lag["LAT"] == lat) &
            (self.agg_lag["LON"] == lon) &
            (self.agg_lag["YEAR_MONTH"] == chosen_month)
        ]

        # If no exact row, fallback to latest available month
        if row.empty:
            row = self.agg_lag[
                (self.agg_lag["LAT"] == lat) &
                (self.agg_lag["LON"] == lon)
            ]
            if row.empty:
                return {"error": "No data for this location"}
            row = row.sort_values("YEAR_MONTH").iloc[-1:]

        # Prepare features
        X_new = row[self.feature_cols].values.reshape(1, -1)
        y_pred = self.model.predict(X_new)[0]

        # Map predictions
        forecast = {}
        for i, step in enumerate([1, 2, 3]):
            forecast[f"month+{step}"] = {
                "T2M": float(y_pred[i*4 + 0]),
                "QV2M": float(y_pred[i*4 + 1]),
                "GWETROOT": float(y_pred[i*4 + 2]),
                "PREC": float(y_pred[i*4 + 3]),
            }
        return forecast
