from dash import Dash, dcc, html
from dash.dependencies import Input, Output

app = Dash(__name__)
app.layout = html.Div(
    [
        html.Div(
            dcc.Upload(
                id="upload-training-data",
                children=html.Div(["Drag and Drop or Select Training Files"]),
                style={
                    "width": "100%",  # Änderung hier
                    "height": "60px",
                    "lineHeight": "60px",
                    "borderWidth": "1px",
                    "borderStyle": "dashed",
                    "borderRadius": "5px",
                    "textAlign": "center",
                    "display": "inline-block",
                    "boxSizing": "border-box",
                },
                multiple=True,
            ),
            style={
                "width": "49%",
                "display": "inline-block",
                "paddingRight": "1%",
            },  # Hinzugefügt
        ),
        html.Div(
            dcc.Upload(
                id="upload-test-data",
                children=html.Div(["Drag and Drop or Select Test Files"]),
                style={
                    "width": "100%",  # Änderung hier
                    "height": "60px",
                    "lineHeight": "60px",
                    "borderWidth": "1px",
                    "borderStyle": "dashed",
                    "borderRadius": "5px",
                    "textAlign": "center",
                    "display": "inline-block",
                    "boxSizing": "border-box",
                },
                multiple=True,
            ),
            style={"width": "49%", "display": "inline-block"},  # Hinzugefügt
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H3("Model Selection"),
                        dcc.RadioItems(
                            options=[
                                {"label": "LGBM", "value": "LGBM"},
                                {"label": "Linear Regression", "value": "LR"},
                            ],
                            value="LGBM",  # Default value
                            id="model-selection-radioitems",
                        ),
                        html.Button(
                            "Upload Config",
                            id="upload-config-btn",
                            style={"display": "none"},
                        ),
                    ],
                    style={"width": "20%", "display": "inline-block"},
                ),
                html.Div(
                    [
                        html.H3("Model Performance"),
                        dcc.Graph(id="model-performance-graph"),
                    ],
                    style={
                        "width": "40%",
                        "display": "inline-block",
                    },
                ),
                html.Div(
                    [
                        html.H3("XAI Method Selection"),
                        dcc.Dropdown(
                            options=[
                                {"label": "SHAP", "value": "SHAP"},
                                {"label": "LIME", "value": "LIME"},
                            ],
                            value="SHAP",  # Default value
                            id="xai-method-dropdown",
                        ),
                    ],
                    style={"width": "40%", "display": "inline-block"},
                ),
            ],
            style={"display": "flex"},
        ),
    ]
)


@app.callback(
    Output("upload-config-btn", "style"), [Input("model-selection-radioitems", "value")]
)
def show_upload_button(selected_model):
    if selected_model == "LGBM":
        return {"display": "inline-block"}
    return {"display": "none"}


if __name__ == "__main__":
    app.run_server(debug=True)
