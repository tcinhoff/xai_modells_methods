from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
from backend.models import LinearRegressionModel
import io
from app_instance import app, PATHS
from .xai_components.coefficients import get_coefficents_component
from .xai_components.pdp import get_pdp_component
from .xai_components.shap import get_shap_component
from .xai_components.lime import get_lime_component
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
                value="Coefficients",  # Standardwert
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
        return "Train a model first."

    point_index = 0 if clickData is None else clickData["points"][0]["pointIndex"]

    test_data = pd.read_csv(PATHS["processed_test_data_path"])
    train_data = pd.read_csv(PATHS["train_data_path"])

    # Anpassung der Aufrufe entsprechend der XAI-Methode
    if xai_method == "Coefficients" and isinstance(model, LinearRegressionModel):
        return get_coefficents_component(test_data.columns)
    elif xai_method == "PDP":
        return get_pdp_component(test_data)
    elif xai_method == "SHAP":
        return get_shap_component(selected_model, point_index, test_data)
    elif xai_method == "LIME":
        return get_lime_component(point_index, train_data, test_data)

    return "Select an appropriate XAI method."


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
    if model is None:
        return "Train a model first."

    if clickData is None:
        return "Click a point in the graph to analyze."

    prediction = pd.read_csv(PATHS["prediction_path"])
    point_index = clickData["points"][0]["pointIndex"]

    return f"Selected point: {prediction.iloc[point_index].date} with prediction {round(prediction.iloc[point_index].yhat, 1)}"
