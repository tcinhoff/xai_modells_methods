from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
from app_instance import app, PATHS
from dash.exceptions import PreventUpdate
import pickle
from ba_xai.configs.xai_methods_config import XAI_METHODS
from .model_selection import MODELS


def get_xai_methods():
    return html.Div(
        [
            html.H3("XAI Method Selection"),
            html.Div(id="selected-point", style={"marginTop": "15px", "marginBottom": "15px"}),
            dcc.Dropdown(
                id="xai-method-dropdown",
            ),
            html.Div(id="xai-output", style={"marginTop": "15px"}),
        ],
        style={"width": "40%", "display": "inline-block"},
    )


@app.callback(
    Output("xai-method-dropdown", "options"),
    [Input("model-selection-radioitems", "value")],
)
def update_xai_dropdown(selected_model):
    if selected_model in MODELS:
        compatible_methods = MODELS[selected_model]["compatible_methods"]
        filtered_options = [
            {"label": XAI_METHODS[method]["label"], "value": method}
            for method in compatible_methods
            if method in XAI_METHODS
        ]
        return filtered_options
    return []


@app.callback(
    Output("xai-output", "children"),
    [
        Input("xai-method-dropdown", "value"),
        Input("model-performance-graph", "clickData"),
    ],
    [State("model-selection-radioitems", "value")],
)
def display_model_evaluation(xai_method, clickData, selected_model):
    if clickData is None or clickData["points"][0]["curveNumber"] != 1:
        raise PreventUpdate

    pickled_model = open(PATHS["model_path"], "rb").read()
    model = pickle.loads(pickled_model)
    if model is None or xai_method is None:
        return None

    point_index = clickData["points"][0]["pointIndex"]

    xai_function = XAI_METHODS[xai_method]["function"]
    return xai_function(selected_model, point_index)


@app.callback(
    Output("selected-point", "children"),
    [
        Input("model-performance-graph", "clickData"),
        Input("xai-method-dropdown", "value"),
    ],
)
def display_selected_point(clickData, selected_method):    
    if selected_method in ["COEFFICIENTS", "PDP"]:
        return "Method is datapoint independent."

    pickled_model = open(PATHS["model_path"], "rb").read()
    model = pickle.loads(pickled_model)
    if model is None or clickData is None or clickData["points"][0]["curveNumber"] != 1:
        return "To use the xai methods train a model and select a datapoint from the prediction."

    prediction = pd.read_csv(PATHS["prediction_path"])
    point_index = clickData["points"][0]["pointIndex"]

    return f"Selected point: {prediction.iloc[point_index].date} with prediction {round(prediction.iloc[point_index].yhat, 1)}"
