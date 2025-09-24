import joblib
import numpy as np

# -------------------------------
# SEASON CLASSIFIER
# -------------------------------
class SeasonClassifier:
    def __init__(self, model_path: str, encoder_path: str):
        self.model = joblib.load(model_path)
        self.encoder = joblib.load(encoder_path)
        self.features = ["LAT", "LON", "DOY", "soil_moisture_7d",
                         "Humidity", "temp_7d", "rain_7d"]

    def predict(self, input_data: dict):
        x = np.array([[input_data[feat] for feat in self.features]])
        pred_class = self.model.predict(x)[0]
        return self.encoder.inverse_transform([pred_class])[0]


# -------------------------------
# IRRIGATION REGRESSOR
# -------------------------------
class IrrigationRegressor:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
        self.features = ["soil_moisture", "Humidity", "T2M", "PRECTOTCORR"]

    def predict(self, input_data: dict):
        x = np.array([[input_data[feat] for feat in self.features]])
        return float(self.model.predict(x)[0])
