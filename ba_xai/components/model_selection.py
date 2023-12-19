from dash import dcc, html
from dash.dependencies import Input, Output
from app_instance import app
from backend.models.models_config import MODELS


def get_model_selection():
    options = [
        {"label": model_info["label"], "value": model}
        for model, model_info in MODELS.items()
    ]

    return html.Div(
        [
            html.H3("Model Selection"),
            dcc.RadioItems(
                options=options,
                value="LR",  
                id="model-selection-radioitems",
            ),
            html.Button(
                "Upload Config",
                id="upload-config-btn",
                style={"display": "none"},
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
