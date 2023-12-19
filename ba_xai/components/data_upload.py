from dash import dcc, html
from dash.dependencies import Input, Output, State, MATCH
import base64
import io
import pandas as pd
from app_instance import app, PATHS


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

    success, message = process_and_save_uploaded_data(contents, filename, id['index'])
    return html.Div([message])


def process_and_save_uploaded_data(contents, filename, id_index):
    if contents is not None and filename.endswith(".csv"):
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        try:
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
            # Entscheiden, welcher Pfad verwendet wird
            path_key = "train_data_path" if id_index == "Training Files" else "test_data_path"
            df.to_csv(PATHS[path_key], index=False)
            return True, f"{filename} - Upload successful"
        except Exception as e:
            return False, "There was an error processing this file."
    else:
        return False, "Please upload a CSV file."