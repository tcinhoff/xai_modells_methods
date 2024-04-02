import json
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from dash.exceptions import PreventUpdate
import base64
import io
from app_instance import app, PATHS
import pickle
from ba_xai.configs.models_config import MODELS
import dash_bootstrap_components as dbc


def get_model_performance():
    return html.Div(
        [
            html.H3("Model Performance"),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "Train Model",
                            id="train-model-button",
                            size="sm",
                            className="align-self-start me-1",
                            style={"marginTop": "15px"},
                        ),
                        width="auto",
                        className="align-self-start",
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                dbc.Button(
                                    "Download Config",
                                    id="download-config-button",
                                    size="sm",
                                    className="align-self-end me-1",
                                    style={"marginTop": "15px"},
                                ),
                                dbc.Button(
                                    "Download Prediction",
                                    id="download-prediction-button",
                                    size="sm",
                                    style={"marginTop": "15px"},
                                ),
                            ],
                        ),
                        width="auto",
                        className="align-self-end me-1",
                    ),
                ],
                justify="between",
                style={"width": "95%"},
            ),
            dcc.Loading(
                id="loading-2",
                type="default",
                children=[
                    dcc.Graph(
                        id="model-performance-graph",
                        config={"staticPlot": False},
                        style={"width": "95%", "height": "400px"},
                    )
                ],
                color="#119DFF",
                fullscreen=False,
            ),
            html.Div(id="train-model-output"),
            dcc.Download(id="download-config"),
            dcc.Download(id="download-prediction"),
        ],
        style={"width": "40%", "display": "inline-block"},
    )


@app.callback(
    Output("download-config", "data"),
    Input("download-config-button", "n_clicks"),
    prevent_initial_call=True,
)
def download_config(n_clicks):
    if n_clicks:
        return dcc.send_file(PATHS["config_path"])


@app.callback(
    Output("download-prediction", "data"),
    Input("download-prediction-button", "n_clicks"),
    prevent_initial_call=True,
)
def download_prediction(n_clicks):
    if n_clicks:
        return dcc.send_file(PATHS["prediction_path"])


@app.callback(
    [
        Output("model-performance-graph", "figure"),
        Output("train-model-output", "children"),
    ],
    [Input("train-model-button", "n_clicks")],
    [
        State("model-selection-radioitems", "value"),
        State("selected-features-store", "data"),
        State("target-column-dropdown", "value"),
    ],
)
def update_graph(n_clicks, selected_model, selected_features, target_col):
    if n_clicks is None:
        raise PreventUpdate

    full_data = pd.read_csv(PATHS["train_test_data_path"])
    train_data = pd.read_csv(PATHS["train_data_path"])
    test_data = pd.read_csv(PATHS["test_data_path"])

    # Überprüfen, ob sowohl Trainings- als auch Testdaten vorhanden sind
    if train_data.empty or test_data.empty:
        return go.Figure(), "Please upload data."

    # Versuche, die Konfiguration zu parsen, falls vorhanden
    model_config = None
    try:
        with open(PATHS["config_path"], "r") as file:
            model_config = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return go.Figure(), f"Error loading config: {e}"

    actual_values = full_data[full_data["date"].isin(test_data["date"])][target_col]
    test_index = test_data.date
    train_data = train_data.drop(columns=["date"])
    test_data = test_data.drop(columns=["date"])

    # reduce to selected features
    if selected_features:
        train_data = train_data[selected_features + [target_col]]
        test_data = test_data[selected_features]

    if selected_model in MODELS:
        model_class = MODELS[selected_model]["class"]
        if model_config is not None and MODELS[selected_model]["config_upload"]:
            model, predictions, std_dev = get_model_prediction(
                model_class(train_data, target_col, model_config), test_data
            )
        else:
            model, predictions, std_dev = get_model_prediction(
                model_class(train_data, target_col), test_data
            )
    else:
        return go.Figure(), "Please select a model."

    predictions_df = pd.DataFrame(
        {"date": test_index, "yhat": predictions, "actual": actual_values.values}
    )
    predictions_plot = create_prediction_plot(predictions_df, std_dev)

    # Speichern des trainierten Modells und der Vorhersagen
    with open(PATHS["model_path"], "wb") as f:
        pickle.dump(model, f)

    pd.DataFrame({"date": test_index, "yhat": predictions}).to_csv(
        PATHS["prediction_path"], index=False
    )
    test_data.to_csv(PATHS["processed_test_data_path"], index=False)
    train_data = train_data.drop(columns=[target_col])
    train_data.to_csv(PATHS["processed_train_data_path"], index=False)

    return predictions_plot, None


def create_prediction_plot(predictions_df, std_dev=None):
    actual_trace = go.Scatter(
        x=predictions_df["date"],
        y=predictions_df["actual"],
        mode="lines",
        name="Actual",
        line=dict(color="red"),
    )

    prediction_trace = go.Scatter(
        x=predictions_df["date"],
        y=predictions_df["yhat"],
        mode="lines",
        name="Predictions",
        line=dict(color="blue"),
    )

    y_val = predictions_df["actual"].values
    y_pred = predictions_df["yhat"].values

    # Entferne Zeilen mit y_val == 0 vor der MAPE-Berechnung
    nonzero_mask = y_val != 0
    y_val = y_val[nonzero_mask]
    y_pred = y_pred[nonzero_mask]

    # Berechne MAPE
    mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100

    layout = go.Layout(
        title="Model Predictions",
        yaxis={"title": "Predicted Value"},
        annotations=[
            dict(
                x=0.5,
                y=-0.25,
                showarrow=False,
                text=f"MAPE: {mape:.2f}%",
                xref="paper",
                yref="paper",
                font=dict(size=16),
            )
        ],
    )

    data = [actual_trace, prediction_trace]

    if std_dev is not None:
        upper_bound = go.Scatter(
            x=predictions_df["date"],
            y=predictions_df["yhat"] + std_dev,
            mode="lines",
            marker=dict(color="#444"),
            line=dict(width=0),
            name="Upper Bound",
            showlegend=False,
            fillcolor="rgba(68, 68, 68, 0.3)",
            fill="tonexty",
        )
        lower_bound = go.Scatter(
            x=predictions_df["date"],
            y=predictions_df["yhat"] - std_dev,
            mode="lines",
            marker=dict(color="#444"),
            line=dict(width=0),
            name="Lower Bound",
            showlegend=False,
            fillcolor="rgba(68, 68, 68, 0.3)",
            fill="tonexty",
        )
        data.extend([lower_bound, upper_bound])

    return {"data": data, "layout": layout}


def get_model_prediction(model, test):
    model.fit()
    output = model.predict(test)
    # Überprüfen, ob die Vorhersage als Tuple mit zwei Elementen (predictions,
    # std_dev) zurückgegeben wurde
    if isinstance(output, tuple) and len(output) == 2:
        predictions, std_dev = output
        return model, predictions, std_dev
    else:
        # Nur Vorhersagen wurden zurückgegeben, keine Standardabweichungen
        predictions = output
        std_dev = None
        return model, predictions, std_dev


def parse_contents(contents):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    return pd.read_csv(io.StringIO(decoded.decode("utf-8")))
