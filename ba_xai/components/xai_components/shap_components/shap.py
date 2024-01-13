from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State
from app_instance import app, PATHS
import pickle
from .shap_pdp import get_shap_pdp_component
from .shap_plot import get_shap_plot_component
from shap.plots import waterfall, bar, beeswarm, violin

selected_shape_method = None


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
                value=selected_shape_method,
            ),
            dcc.Loading(
                id="loading-1",
                type="dot",
                children=html.Div(id="shap-output", style={"marginTop": "15px"}),
                style={"marginTop": "35px"},
            ),
        ],
        style={"width": "100%", "display": "inline-block"},
    )


@app.callback(
    Output("shap-output", "children"),
    [
        Input("shap-method-dropdown", "value"),
        Input("model-performance-graph", "clickData"),
    ],
    State("model-selection-radioitems", "value"),
)
def display_model_evaluation(shap_method, clickData, selected_model):
    global selected_shape_method
    if selected_model is None or shap_method is None:
        return html.Div()

    if callback_context.triggered[0]["prop_id"] == "shap-method-dropdown.value":
        selected_shape_method = shap_method

    point_index = 0 if clickData is None else clickData["points"][0]["pointIndex"]
    if selected_shape_method == "Shap-PDP":
        return get_shap_pdp_component()
    return get_shap_plot_component(
        selected_model, point_index, SHAP_METHODS[selected_shape_method]
    )


SHAP_METHODS = {
    "SHAP-Waterfall": {
        "label": "SHAP Waterfall",
        "local": True,
        "function": waterfall,
    },
    "Shap-PDP": {
        "label": "SHAP PDP",
        "local": None,
        "function": None,
    },
    "SHAP-Bar": {
        "label": "SHAP Bar",
        "local": True,
        "function": bar,
    },
    "SHAP-Global-Bar": {
        "label": "SHAP Global Bar",
        "local": False,
        "function": bar,
    },
    "SHAP-Beeswarm": {
        "label": "SHAP Beeswarm",
        "local": False,
        "function": beeswarm,
    },
    "SHAP-Violin": {
        "label": "SHAP Violin",
        "local": False,
        "function": violin,
    },
}
