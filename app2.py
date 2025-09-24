from flask import Flask, request, jsonify
from weather import WeatherForecaster
from static_models import SeasonClassifier, IrrigationRegressor  # file name fixed

# ----------------------------
# 1. Load models once
# ----------------------------
feature_cols = ['LAT','LON','month_sin','month_cos'] + \
               [f'{v}_lag{l}' for v in ['T2M','QV2M','GWET','PREC'] for l in [1,2,3]]

# Load wrappers
wf = WeatherForecaster("weather_3month_model.pkl", feature_cols, "agg_lag.csv")
season_model = SeasonClassifier("season_classifier.pkl", "season_labelencoder.pkl")
irrigation_model = IrrigationRegressor("irrigation_regressor.pkl")

# ----------------------------
# 2. Create Flask app
# ----------------------------
app = Flask(__name__)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    Input JSON example:
    {
        "LAT": 18.0,
        "LON": 75.0,
        "YEAR_MONTH": "2025-04",
        "features": {
            "DOY": 120,
            "soil_moisture": 0.45,
            "Humidity": 65,
            "T2M": 27.0,
            "PRECTOTCORR": 1.2,
            "soil_moisture_7d": 0.45,
            "temp_7d": 26.7,
            "rain_7d": 12.4
        }
    }
    """

    # data = request.get_json()
    data ={
        "LAT": 18.0,
        "LON": 75.0,
        "YEAR_MONTH": "2025-04",
        "features": {
            "DOY": 120,
            "soil_moisture": 0.45,
            "Humidity": 65,
            "T2M": 27.0,
            "PRECTOTCORR": 1.2,
            "soil_moisture_7d": 0.45,
            "temp_7d": 26.7,
            "rain_7d": 12.4
        }
    }

    lat = data["LAT"]
    lon = data["LON"]
    chosen_month = data["YEAR_MONTH"]
    features = data["features"]

    # ----------------------------
    # (a) Time-series forecast
    # ----------------------------
    forecast = wf.forecast(chosen_month=chosen_month, lat=lat, lon=lon)

    # ----------------------------
    # (b) Season classifier
    # ----------------------------
    input_season = {
        "LAT": lat,
        "LON": lon,
        "DOY": features["DOY"],
        "soil_moisture_7d": features["soil_moisture_7d"],
        "Humidity": features["Humidity"],
        "temp_7d": features["temp_7d"],
        "rain_7d": features["rain_7d"]
    }
    season = season_model.predict(input_season)

    # ----------------------------
    # (c) Irrigation regressor
    # ----------------------------
    input_irrigation = {
        "soil_moisture": features["soil_moisture"],
        "Humidity": features["Humidity"],
        "T2M": features["T2M"],
        "PRECTOTCORR": features["PRECTOTCORR"]
    }
    irrigation = irrigation_model.predict(input_irrigation)

    # ----------------------------
    # (d) Combine results
    # ----------------------------
    result = {
        "time_series_forecast": forecast,
        "irrigation": irrigation,
        "season": season
    }
    print(result)
    return jsonify(result)

# ----------------------------
# 3. Run app
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
