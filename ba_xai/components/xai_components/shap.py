from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
from app_instance import app, PATHS
import pickle
from .shap_components.shap_config import SHAP_METHODS


def get_shap_component(selected_model, point_index):
    options = [
        {"label": method_info["label"], "value": method}
        for method, method_info in SHAP_METHODS.items()
    ]

    return html.Div(
        [
               dcc.Dropdown(
                options=options,
                id="shap-method-dropdown",
            ),
            html.Div(id="shap-output", style={"marginTop": "15px"}),
        ],
        style={"width": "100%", "display": "inline-block"},
    )


@app.callback(
    Output("shap-output", "children"),
    [
        Input("shap-method-dropdown", "value"),
        Input("model-performance-graph", "clickData"),
    ],
    [State("model-selection-radioitems", "value")],
)
def display_model_evaluation(shap_method, clickData, selected_model):
    pickled_model = open(PATHS["model_path"], "rb").read()
    model = pickle.loads(pickled_model)
    if model is None or shap_method is None:
        return None

    point_index = 0 if clickData is None else clickData["points"][0]["pointIndex"]
    return SHAP_METHODS[shap_method]["function"](selected_model, point_index)

