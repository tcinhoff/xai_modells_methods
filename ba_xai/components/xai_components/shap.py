from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State
from app_instance import app
from dash.exceptions import PreventUpdate
from .shap_pdp import get_shap_pdp_component
from ba_xai.backend.xai_methods.shap import SHAP
from ba_xai.configs.shap_config import SHAP_METHODS

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
                id="loading-4",
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
    if clickData is None or clickData["points"][0]["curveNumber"] != 1:
        raise PreventUpdate
        
    global selected_shape_method
    if selected_model is None or shap_method is None:
        return html.Div()

    if callback_context.triggered[0]["prop_id"] == "shap-method-dropdown.value":
        selected_shape_method = shap_method

    point_index = clickData["points"][0]["pointIndex"]
    if selected_shape_method == "Shap-PDP":
        return get_shap_pdp_component()
    return SHAP.get_shap_plot_component(
        selected_model, point_index, SHAP_METHODS[selected_shape_method]
    )

