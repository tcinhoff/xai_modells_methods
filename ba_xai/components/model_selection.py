import json
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from app_instance import PATHS, app
from backend.models.models_config import MODELS
from components.data_upload import get_data_upload_button
from .model_config_selection import get_model_config_selection
from .feature_selection_modal import get_feature_selection_modal


def get_data_upload_model_selection():
    options = [
        {"label": model_info["label"], "value": model}
        for model, model_info in MODELS.items()
    ]

    return html.Div(
        [
            html.H3("Data Upload"),
            get_data_upload_button(),
            get_feature_selection_modal(),
            html.H3("Model Selection", style={"marginTop": "30px"}),
            dbc.RadioItems(
                options=options,
                value="LR",
                id="model-selection-radioitems",
            ),
            html.Div(id="model-config-selection-div"),
        ],
        style={"width": "20%", "display": "inline-block"},
    )


@app.callback(
    Output("model-config-selection-div", "children"),
    [Input("model-selection-radioitems", "value")],
)
def update_model_parameters(selected_model):
    with open(PATHS["config_path"], "w") as file:
        json.dump(None, file, indent=4)

    if MODELS[selected_model]["config_upload"]:
        return get_model_config_selection()
    return html.Div()

