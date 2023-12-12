from dash import Dash
import pickle

model_path = "model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(None, f)

app = Dash(__name__, suppress_callback_exceptions=True)

