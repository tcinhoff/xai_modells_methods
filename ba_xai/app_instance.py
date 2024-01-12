from dash import Dash
import pickle
import dash_bootstrap_components as dbc

PATHS = {
    "model_path": "ba_xai/temp/model.pkl",
    "train_data_path": "ba_xai/temp/train_data.csv",
    "processed_train_data_path": "ba_xai/temp/processed_train_data.csv",
    "test_data_path": "ba_xai/temp/test_data.csv",
    "processed_test_data_path": "ba_xai/temp/processed_test_data.csv",
    "prediction_path": "ba_xai/temp/prediction.csv",
    "path_to_scaler": "ba_xai/temp/scaler.pkl",
    "config_path": "ba_xai/temp/config.json",
}

for path in PATHS.values():
    with open(path, "wb") as f:
        pickle.dump(None, f)


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
