from dash import Dash
import pickle
import dash_bootstrap_components as dbc
import os

PATHS = {
    "model_path": "temp/model.pkl",
    "train_test_data_path": "temp/train_test_data.csv",
    "train_data_path": "temp/train_data.csv",
    "processed_train_data_path": "temp/processed_train_data.csv",
    "test_data_path": "temp/test_data.csv",
    "processed_test_data_path": "temp/processed_test_data.csv",
    "prediction_path": "temp/prediction.csv",
    "path_to_scaler": "temp/scaler.pkl",
    "config_path": "temp/config.json",
}

for path in PATHS.values():
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(None, f)


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
