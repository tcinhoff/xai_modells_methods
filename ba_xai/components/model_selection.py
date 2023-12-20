from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from app_instance import app
from backend.models.models_config import MODELS
from components.data_upload import get_data_upload_button


def get_data_upload_model_selection():
    options = [
        {"label": model_info["label"], "value": model}
        for model, model_info in MODELS.items()
    ]

    return html.Div(
        [
            html.H3("Data Upload"),
            get_data_upload_button("Training Files"),
            get_data_upload_button("Test Files"),
            html.H3("Model Selection", style={"marginTop": "30px"}),
            dbc.RadioItems(
                options=options,
                value="LR",
                id="model-selection-radioitems",
            ),
            html.Div(
                dbc.Button(
                    "Upload Config",
                    id="upload-config-btn",
                    style={"display": "none"},
                    size="sm",
                ),
                style={"width": "100%", "marginTop": "15px"},
            ),
        ],
        style={"width": "20%", "display": "inline-block"},
    )


@app.callback(
    Output("upload-config-btn", "style"), [Input("model-selection-radioitems", "value")]
)
def show_upload_config_button(selected_model):
    if MODELS[selected_model]["config_upload"]:
        return {"display": "inline-block"}
    return {"display": "none"}
