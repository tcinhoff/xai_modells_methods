from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
from app_instance import app, PATHS
import pickle
from .xai_components.xai_methods_config import XAI_METHODS


def get_xai_methods():
    options = [
        {"label": method_info["label"], "value": method}
        for method, method_info in XAI_METHODS.items()
    ]

    return html.Div(
        [
            html.H3("XAI Method Selection"),
            dcc.Dropdown(
                options=options,
                value="Coefficients",
                id="xai-method-dropdown",
            ),
            html.Div(id="selected-point", style={"marginTop": "15px"}),
            html.Div(id="xai-output", style={"marginTop": "15px"}),
        ],
        style={"width": "40%", "display": "inline-block"},
    )


@app.callback(
    Output("xai-output", "children"),
    [
        Input("xai-method-dropdown", "value"),
        Input("model-performance-graph", "clickData"),
    ],
    [State("model-selection-radioitems", "value")],
)
def display_model_evaluation(xai_method, clickData, selected_model):
    pickled_model = open(PATHS["model_path"], "rb").read()
    model = pickle.loads(pickled_model)
    if model is None:
        return "Train a model first or select an datapoint."

    point_index = 0 if clickData is None else clickData["points"][0]["pointIndex"]

    if xai_method in XAI_METHODS and isinstance(
        model, tuple(XAI_METHODS[xai_method]["compatible_models"])
    ):
        xai_function = XAI_METHODS[xai_method]["function"]
        return xai_function(selected_model, point_index)
    else:
        return "Select an appropriate XAI method or ensure the model is compatible."


@app.callback(
    Output("selected-point", "children"),
    [
        Input("model-performance-graph", "clickData"),
        Input("xai-method-dropdown", "value"),
    ],
)
def display_selected_point(clickData, selected_method):
    if selected_method in ["Coefficients", "PDP"]:
        return None

    pickled_model = open(PATHS["model_path"], "rb").read()
    model = pickle.loads(pickled_model)
    if model is None or clickData is None:
        return None

    prediction = pd.read_csv(PATHS["prediction_path"])
    point_index = clickData["points"][0]["pointIndex"]

    return f"Selected point: {prediction.iloc[point_index].date} with prediction {round(prediction.iloc[point_index].yhat, 1)}"
