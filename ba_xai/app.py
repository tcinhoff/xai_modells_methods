from app_instance import app
from dash import html, dcc
from components.model_selection import get_data_upload_model_selection
from components.model_performance import get_model_performance
from components.xai_methods import get_xai_methods

app.layout = html.Div(
    [
        html.H1("Explainable AI Methods for Time Series Forecasting", style={"textAlign": "center", "marginBottom": "30px"}),
        html.Div(
            [
                get_data_upload_model_selection(),
                get_model_performance(),
                get_xai_methods(),
            ],
            style={"display": "flex", "marginTop": "20px"},
        ),
    ],
    style={"margin": "20px"},
)

if __name__ == "__main__":
    app.run_server(debug=True)
