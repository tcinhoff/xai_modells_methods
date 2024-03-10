import base64
import json
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from app_instance import app, PATHS
from .model_paramters_selection import generate_model_parameters
from ba_xai.configs.models_config import MODELS


def get_model_config_selection():
    return html.Div(
        [
            html.H3("Model Configuration", style={"marginTop": "30px"}),
            dbc.RadioItems(
                id="config-method-toggle",
                options=[
                    {"label": "upload config", "value": "upload"},
                    {"label": "manually configure", "value": "manual"},
                ],
                value="upload",
                inline=True,
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
                style={"width": "100%", "marginTop": "15px"},
            ),
            html.Div(id="manually-configure-div"),
        ],
    )


@app.callback(
    [
        Output("manually-configure-div", "children"),
        Output("upload-config", "style"),
        Output("upload-config", "children"),
    ],
    [
        Input("config-method-toggle", "value"),
        Input("model-selection-radioitems", "value"),
        Input("upload-config", "contents"),
    ],
    State("upload-config", "filename"),
)
def update_model_parameters(selected_method, selected_model, contents, filename):
    config_upload_style = {
        "width": "80%",
        "borderWidth": "1px",
        "borderStyle": "dashed",
        "borderRadius": "5px",
        "textAlign": "center",
        "display": "inline-block",
        "boxSizing": "border-box",
        "marginTop": "15px",
        "padding": "3px",
    }
    if selected_method == "manual":
        return generate_model_parameters(selected_model), {"display": "none"}, None

    if contents is None:
        with open(PATHS["config_path"], "w") as file:
            json.dump(None, file, indent=4)
        return None, config_upload_style, html.Div([f"Drag and Drop or Select Config"])

    return (
        None,
        config_upload_style,
        html.Div([process_and_save_uploaded_config(contents, filename)]),
    )


def process_and_save_uploaded_config(contents, filename):
    # test if file is json and if it is a valid config
    if contents is not None and filename.endswith(".json"):
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        try:
            config = json.loads(decoded.decode("utf-8"))
            with open(PATHS["config_path"], "w") as file:
                json.dump(config, file, indent=4)
            return f"{filename} - Upload successful"
        except Exception as e:
            print(e)
            return "There was an error processing this file."
