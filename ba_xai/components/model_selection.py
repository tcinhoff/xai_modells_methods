import base64
import io
import json
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd
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
                id="upload-config-div",
                children=dcc.Upload(
                    id="upload-config",
                    children=html.Div([f"Drag and Drop or Select Config"]),
                    style={
                        "width": "80%",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",
                        "display": "inline-block",
                        "boxSizing": "border-box",
                        "marginTop": "15px",
                        "padding": "3px",
                    },
                    multiple=False,
                ),
                style={"width": "100%", "marginTop": "15px", "display": "none"},
            ),
        ],
        style={"width": "20%", "display": "inline-block"},
    )


@app.callback(
    [Output("upload-config-div", "style"),
     Output("upload-config", "contents")],
     [Input("model-selection-radioitems", "value")]
)
def show_upload_config_button(selected_model):
    if MODELS[selected_model]["config_upload"]:
        return {"display": "inline-block"}, None
    return {"display": "none"}, None


@app.callback(
    Output("upload-config", "children"),
    Input("upload-config", "contents"),
    State("upload-config", "filename"),
)
def update_upload_text(contents, filename):
    if contents is None:
        return html.Div([f"Drag and Drop or Select Config"])

    return html.Div([process_and_save_uploaded_config(contents, filename)])


def process_and_save_uploaded_config(contents, filename):
    # test if file is json and if it is a valid config 
    if contents is not None and filename.endswith(".json"):
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        try:
            config = json.loads(decoded.decode("utf-8"))
            return f"{filename} - Upload successful"
        except Exception as e:
            return "There was an error processing this file."
    
