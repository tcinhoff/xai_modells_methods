from dash import dcc, html
from dash.dependencies import Input, Output, State, MATCH
import base64
import io
import pandas as pd
from app_instance import app


def get_data_upload_button(id):
    return html.Div(
        dcc.Upload(
            id={"type": "upload-data", "index": id},
            children=html.Div([f"Drag and Drop or Select {id}"]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "display": "inline-block",
                "boxSizing": "border-box",
            },
            multiple=False,
        ),
        style={
            "width": "49%",
            "display": "inline-block",
            "paddingRight": "1%",
        },
    )


@app.callback(
    Output({"type": "upload-data", "index": MATCH}, "children"),
    [Input({"type": "upload-data", "index": MATCH}, "contents")],
    [
        State({"type": "upload-data", "index": MATCH}, "filename"),
        State({"type": "upload-data", "index": MATCH}, "id"),
    ],
)
def update_upload_text(contents, filename, id):
    if contents is None:
        return html.Div([f"Drag and Drop or Select {id['index']}"])

    _, processed_data = process_uploaded_data(contents, filename)
    if processed_data is not None:
        return html.Div([f"{filename} - Upload successful"])
    else:
        return html.Div(["There was an error processing this file."])


def process_uploaded_data(contents, filename):
    if contents is not None:
        # Überprüfen Sie, ob die Datei eine CSV-Datei ist
        if filename.endswith(".csv"):
            content_type, content_string = contents.split(",")
            decoded = base64.b64decode(content_string)
            try:
                df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
                return html.Div(
                    [f"{filename} - Drag and Drop or Select for changing"]
                ), df.to_json(date_format="iso", orient="split")
            except Exception as e:
                return html.Div(["There was an error processing this file."]), None
        else:
            return html.Div(["Please upload a CSV file."]), None
    return None, None


@app.callback(
    Output("hidden-div-for-train-data", "children"),
    Output("hidden-div-for-test-data", "children"),
    [
        Input({"type": "upload-data", "index": "Training Files"}, "contents"),
        Input({"type": "upload-data", "index": "Test Files"}, "contents"),
    ],
    [
        State({"type": "upload-data", "index": "Training Files"}, "filename"),
        State({"type": "upload-data", "index": "Test Files"}, "filename"),
    ],
)
def update_hidden_divs(train_contents, test_contents, train_filename, test_filename):
    if train_contents:
        _, train_data = process_uploaded_data(train_contents, train_filename)
    else:
        train_data = None

    if test_contents:
        _, test_data = process_uploaded_data(test_contents, test_filename)
    else:
        test_data = None

    return train_data, test_data
