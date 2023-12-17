from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
from backend.models import LinearRegressionModel
import io
from app_instance import app, model_path
from .xai_components.coefficients import get_coefficents_component
from .xai_components.pdp import get_pdp_component
from .xai_components.shap import get_shap_component
from .xai_components.lime import get_lime_component
import pickle


def get_xai_methods():
    return html.Div(
        [
            html.H3("XAI Method Selection"),
            dcc.Dropdown(
                options=[
                    {
                        "label": "Modellkoeffizienten",
                        "value": "Coefficients",
                    },
                    {"label": "Partial Dependence Plots", "value": "PDP"},
                    {"label": "LIME", "value": "LIME"},
                    {"label": "SHAP", "value": "SHAP"},
                ],
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
        Input("hidden-div-for-prediction", "children"),
    ],
    [
        State("model-selection-radioitems", "value"),
        State("hidden-div-for-processed-test-data", "children"),
        State("hidden-div-for-train-data", "children")
    ],
)
def display_model_evaluation(xai_method, clickData, _, selected_model, test_data_json, train_data_json):
    pickled_model = open(model_path, "rb").read()
    model = pickle.loads(pickled_model)
    if model is None or test_data_json is None:
        return "Train a model first and ensure data is loaded."

    point_index = 0 if clickData is None else clickData["points"][0]["pointIndex"]
    test_data = pd.read_json(io.StringIO(test_data_json), orient="split")
    train_data = pd.read_json(io.StringIO(train_data_json), orient="split")
    train_data = train_data.drop(columns=["date"])

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
    State("hidden-div-for-prediction", "children"),
)
def display_selected_point(clickData, selected_method, prediction_json):
    if selected_method in ["Coefficients", "PDP"]:
        return None
    pickled_model = open(model_path, "rb").read()
    model = pickle.loads(pickled_model)
    if model is None:
        return "Train a model first."

    if clickData is None:
        return "Click a point in the graph to analyze."

    point_index = clickData["points"][0]["pointIndex"]
    prediction = pd.read_json(io.StringIO(prediction_json), orient="split")
    return f"Selected point: {prediction.iloc[point_index].date} with prediction {round(prediction.iloc[point_index].yhat, 1)}"
