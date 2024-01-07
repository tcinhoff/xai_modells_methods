import json
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
from dash.exceptions import PreventUpdate
import base64
import io
from app_instance import app, PATHS
import pickle
from backend.models.models_config import MODELS
import dash_bootstrap_components as dbc


def get_model_performance():
    return html.Div(
        [
            html.H3("Model Performance"),
            dbc.Button("Train Model", id="train-model-button", size="sm"),
            dcc.Graph(
                id="model-performance-graph",
                config={"staticPlot": False},
                style={"width": "95%", "height": "400px"},
            ),
            html.Div(id="train-model-output"),
        ],
        style={"width": "40%", "display": "inline-block"},
    )


@app.callback(
    [
        Output("model-performance-graph", "figure"),
        Output("train-model-output", "children"),
    ],
    [Input("train-model-button", "n_clicks")],
    [State("model-selection-radioitems", "value"),
     State("upload-config", "contents")],
)
def update_graph(n_clicks, selected_model, model_config_json):
    if n_clicks is None:
        raise PreventUpdate

    train_data = pd.read_csv(PATHS["train_data_path"])
    test_data = pd.read_csv(PATHS["test_data_path"])

    # Überprüfen, ob sowohl Trainings- als auch Testdaten vorhanden sind
    if train_data.empty or test_data.empty:
        return go.Figure(), "Please upload data."

    # Versuche, die Konfiguration zu parsen, falls vorhanden
    model_config = None
    if model_config_json:
        try:
            content_type, content_string = model_config_json.split(',')
            decoded = base64.b64decode(content_string)
            model_config = json.loads(decoded.decode('utf-8'))
        except (ValueError, json.JSONDecodeError) as e:
            return go.Figure(), f"Error parsing JSON: {e}"
        
    test_index = test_data.date
    train_data = train_data.drop(columns=["date"])
    test_data = test_data.drop(columns=["date"])
    target_col = list(set(train_data.columns) - set(test_data.columns))[0]

    if selected_model in MODELS:
        model_class = MODELS[selected_model]["class"]
        if model_config is not None and MODELS[selected_model]["config_upload"]:
            model, predictions = get_model_prediction(
                model_class(train_data, target_col, model_config), test_data
            )
        else:
            model, predictions = get_model_prediction(model_class(train_data, target_col), test_data)
    else:
        return go.Figure(), "Please select a model."

    predictions_df = pd.DataFrame({"date": test_index, "yhat": predictions})
    predictions_plot = create_prediction_plot(predictions_df)

    # Speichern des trainierten Modells und der Vorhersagen
    with open(PATHS["model_path"], "wb") as f:
        pickle.dump(model, f)

    predictions_df.to_csv(PATHS["prediction_path"], index=False)
    test_data.to_csv(PATHS["processed_test_data_path"], index=False)
    train_data = train_data.drop(columns=["yhat"])
    train_data.to_csv(PATHS["processed_train_data_path"], index=False)

    return predictions_plot, None


def create_prediction_plot(predictions_df):
    trace = go.Scatter(
        x=predictions_df["date"],
        y=predictions_df["yhat"],
        mode="lines",
        name="Predictions",
    )
    layout = go.Layout(
        title="Model Predictions",
        xaxis={"title": "Date"},
        yaxis={"title": "Predicted Value"},
    )
    return {"data": [trace], "layout": layout}


def get_model_prediction(model, test):
    model.fit()
    predictions = model.predict(test)
    return model, predictions


def parse_contents(contents):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    return pd.read_csv(io.StringIO(decoded.decode("utf-8")))
