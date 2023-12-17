from dash import dcc, html
from dash.dependencies import Input, Output
from app_instance import app


def get_model_selection():
    return html.Div(
        [
            html.H3("Model Selection"),
            dcc.RadioItems(
                options=[
                    {"label": "Linear Regression", "value": "LR"},
                    {"label": "GAM", "value": "GAM"},
                    {"label": "LGBM", "value": "LGBM"},
                    {"label": "XGBoost", "value": "XGBoost"},
                ],
                value="LR",  # Default value
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
    if selected_model in ["LGBM", "XGBoost", "GAM"]:
        return {"display": "inline-block"}
    return {"display": "none"}
