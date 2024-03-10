from datetime import datetime, timedelta
from dash import dcc, html
from dash.dependencies import Input, Output, State
import base64
import io
import pandas as pd
from app_instance import app, PATHS


def get_data_upload_button():
    return html.Div(
        [
            dcc.Upload(
                id="upload-data",
                children=html.Div([f"Upload Train-Test-Data"]),
                style={
                    "width": "76%",
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
            html.H5("Select Trainingsrange", style={"marginTop": "15px"}),
            dcc.DatePickerRange(
                id={"type": "date-picker-range", "index": "train"},
                start_date_placeholder_text="Start Date",
                end_date_placeholder_text="End Date",
                display_format="YYYY-MM-DD",
                style={"width": "100%", "display": "inline-block"},
            ),
            html.H5("Select Testrange", style={"marginTop": "10px"}),
            dcc.DatePickerRange(
                id={"type": "date-picker-range", "index": "test"},
                start_date_placeholder_text="Start Date",
                end_date_placeholder_text="End Date",
                display_format="YYYY-MM-DD",
                style={"width": "100%", "display": "inline-block"},
            ),
            dcc.Dropdown(
                id='target-column-dropdown',
                options=[],  # Options will be set dynamically after data upload
                placeholder="Select a target column",
                style={"width": "87.4%", "display": "inline-block", "marginTop": "10px"},
            ),
            html.Div(id='dataset-split-status', style={"marginTop": "10px"}),
            html.Div(id="date-range-info", style={"display": "none"}),
        ],
        style={"width": "100%", "display": "inline-block"},
    )


@app.callback(
    Output("upload-data", "children"),
    Input("upload-data", "contents"),
    [
        State("upload-data", "filename"),
    ],
)
def update_upload_text(contents, filename):
    if contents is None:
        return html.Div([f"Upload Train-Test-Data"])

    return html.Div([process_and_save_uploaded_data(contents, filename)])


def process_and_save_uploaded_data(contents, filename):
    if contents is not None and filename.endswith(".csv"):
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        try:
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
            df.to_csv(PATHS["train_test_data_path"], index=False)
            return f"{filename} - Upload successful"
        except Exception as e:
            return "There was an error processing this file."
    else:
        return "Please upload a CSV file."


@app.callback(
    Output("date-range-info", "children"),
    [Input("upload-data", "contents")],
    [State("upload-data", "filename")],
)
def update_date_range_info(contents, filename):
    if contents is not None:
        # Dateiinhalt verarbeiten
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        df["date"] = pd.to_datetime(df["date"])  # Annahme: 'date' ist die Datums-Spalte
        min_date = df["date"].min().date()
        max_date = df["date"].max().date()
        return f"{min_date},{max_date}"
    return ""


# Callback, um die DatePickerRange-Komponenten basierend auf den extrahierten Daten zu aktualisieren
@app.callback(
    [
        Output({'type': 'date-picker-range', 'index': 'train'}, 'min_date_allowed'),
        Output({'type': 'date-picker-range', 'index': 'train'}, 'max_date_allowed'),
        Output({'type': 'date-picker-range', 'index': 'train'}, 'start_date'),
        Output({'type': 'date-picker-range', 'index': 'train'}, 'end_date'),
        Output({'type': 'date-picker-range', 'index': 'test'}, 'min_date_allowed'),
        Output({'type': 'date-picker-range', 'index': 'test'}, 'max_date_allowed'),
        Output({'type': 'date-picker-range', 'index': 'test'}, 'start_date'),
        Output({'type': 'date-picker-range', 'index': 'test'}, 'end_date'),
    ],
    [Input('date-range-info', 'children')]
)
def set_date_picker_ranges(date_range_info):
    if ',' in date_range_info:
        min_date_str, max_date_str = date_range_info.split(',')
        min_date = datetime.strptime(min_date_str, '%Y-%m-%d')
        max_date = datetime.strptime(max_date_str, '%Y-%m-%d')

        # Ermitteln des letzten vollständigen Monats innerhalb des Datumsbereichs des Datensatzes
        if max_date.day == (max_date.replace(day=28) + timedelta(days=4)).day - 1:
            # Max_date ist der letzte Tag des Monats
            last_full_month_end = max_date
        else:
            # Gehe zum ersten Tag des Monats und dann einen Monat zurück
            last_full_month_end = max_date.replace(day=1) - timedelta(days=1)

        last_full_month_start = last_full_month_end.replace(day=1)

        # Training range endet am Tag vor dem Start des letzten vollständigen Monats
        train_range_end = last_full_month_start - timedelta(days=1)
        
        return [
            min_date.strftime('%Y-%m-%d'),  # Min Trainingsdatum
            max_date.strftime('%Y-%m-%d'),  # Max Trainingsdatum
            min_date.strftime('%Y-%m-%d'),  # Start Trainingsdatum
            train_range_end.strftime('%Y-%m-%d'),  # Ende Trainingsdatum
            min_date.strftime('%Y-%m-%d'),  # Min Testdatum
            max_date.strftime('%Y-%m-%d'),  # Max Testdatum
            last_full_month_start.strftime('%Y-%m-%d'),  # Start Testdatum
            last_full_month_end.strftime('%Y-%m-%d'),  # Ende Testdatum
        ]
    return [None] * 8  # Fallback, falls keine Daten vorhanden sind

@app.callback(
    Output('dataset-split-status', 'children'),  # Ein Ausgabeelement für Statusmeldungen
    [
        Input({'type': 'date-picker-range', 'index': 'train'}, 'start_date'),
        Input({'type': 'date-picker-range', 'index': 'train'}, 'end_date'),
        Input({'type': 'date-picker-range', 'index': 'test'}, 'start_date'),
        Input({'type': 'date-picker-range', 'index': 'test'}, 'end_date'),
        Input('target-column-dropdown', 'value'),
    ],
    [
        State('date-range-info', 'children'),  # Zustand der Datumsspanne aus dem Upload
    ]
)
def split_and_save_datasets(train_start, train_end, test_start, test_end, target_column, date_range_info):
    if not all([train_start, train_end, test_start, test_end, target_column]):
        return 'Please complete the date ranges.'

    if ',' in date_range_info:
        # Laden des Gesamtdatasets
        min_date_str, max_date_str = date_range_info.split(',')
        df = pd.read_csv(PATHS["train_test_data_path"])  # Angenommen, dies ist der Pfad zum Gesamtdataset
        df['date'] = pd.to_datetime(df['date'])

        # Aufteilung in Trainings- und Testdaten
        train_df = df[(df['date'] >= pd.to_datetime(train_start)) & (df['date'] <= pd.to_datetime(train_end))]
        test_df = df[(df['date'] >= pd.to_datetime(test_start)) & (df['date'] <= pd.to_datetime(test_end))]

        # Speichern der aufgeteilten Daten
        train_df.to_csv(PATHS["train_data_path"], index=False)
        test_df.drop(columns=[target_column]).to_csv(PATHS["test_data_path"], index=False)  # Entfernen der Zielvariable

        return 'Successfully split and saved Train-Test-Data.'
    return 'Please upload a Dataset.'

@app.callback(
    Output('target-column-dropdown', 'options'),
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_target_column_dropdown(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        return [{'label': column, 'value': column} for column in df.columns]
    return []
