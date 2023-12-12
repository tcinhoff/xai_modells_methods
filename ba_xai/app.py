# Importe und gobale Variablen

from app_instance import app
from dash import html
from components.data_upload import get_data_upload_button
from components.model_selection import get_model_selection
from components.model_performance import get_model_performance
from components.xai_methods import get_xai_methods

app.layout = html.Div(
    [
        get_data_upload_button("Training Files"),
        get_data_upload_button("Test Files"),
        html.Div(
            [
                get_model_selection(),
                get_model_performance(),
                get_xai_methods(),
            ],
            style={"display": "flex"},
        ),
        # Verborgene Divs für die Speicherung von Daten
        html.Div(id="hidden-div-for-train-data", style={"display": "none"}),
        html.Div(id="hidden-div-for-test-data", style={"display": "none"}),
        html.Div(id="hidden-div-for-processed-test-data", style={"display": "none"}),
        html.Div(id="hidden-div-for-prediction", style={"display": "none"}),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
