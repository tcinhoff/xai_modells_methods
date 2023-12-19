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


def get_model_performance():
    return html.Div(
        [
            html.H3("Model Performance"),
            html.Button("Train Model", id="train-model-button"),
            dcc.Graph(
                id="model-performance-graph",
                config={"staticPlot": False},
                style={"width": "100%", "height": "400px"},
            ),
            html.Div(id="train-model-output"),
        ],
        style={"width": "40%", "display": "inline-block"},
    )


@app.callback(
    [
        Output("model-performance-graph", "figure"),
        Output("train-model-output", "children"),
        Output("hidden-div-for-processed-test-data", "children"),
        Output("hidden-div-for-prediction", "children"),
    ],
    [Input("train-model-button", "n_clicks")],
    [
        State("hidden-div-for-train-data", "children"),
        State("hidden-div-for-test-data", "children"),
        State("model-selection-radioitems", "value"),
    ],
)
def update_graph(n_clicks, train_data_json, test_data_json, selected_model):
    if n_clicks is None:
        raise PreventUpdate

    # Überprüfen, ob sowohl Trainings- als auch Testdaten vorhanden sind
    if not train_data_json or not test_data_json:
        return go.Figure(), html.Div("Please upload data."), None, None

    # Datenverarbeitung
    train_df = pd.read_json(io.StringIO(train_data_json), orient="split")
    test_df = pd.read_json(io.StringIO(test_data_json), orient="split")
    test_index = test_df.date
    train_df = train_df.drop(columns=["date"])
    test_df = test_df.drop(columns=["date"])

    if selected_model in MODELS:
        model_class = MODELS[selected_model]["class"]
        model, predictions = get_model_prediction(model_class(train_df), test_df)
    else:
        return go.Figure(), "Please select a model.", None, None

    predictions_df = pd.DataFrame({"date": test_index, "yhat": predictions})

    with open(PATHS["model_path"], "wb") as f:
        pickle.dump(model, f)

    predictions_plot = create_prediction_plot(predictions_df)

    return (
        predictions_plot,
        None,
        test_df.to_json(date_format="iso", orient="split"),
        predictions_df.to_json(date_format="iso", orient="split"),
    )


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
