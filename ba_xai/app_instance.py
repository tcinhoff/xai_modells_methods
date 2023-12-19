from dash import Dash
import pickle

PATHS = {
    "model_path": "ba_xai/temp/model.pkl",
    "train_data_path": "ba_xai/temp/train_data.csv",
    "test_data_path": "ba_xai/temp/test_data.csv",
    "processed_test_data_path": "ba_xai/temp/processed_test_data.csv",
    "prediction_path": "ba_xai/temp/prediction.csv",
}
for path in PATHS.values():
    with open(path, "wb") as f:
        pickle.dump(None, f)

app = Dash(__name__, suppress_callback_exceptions=True)
