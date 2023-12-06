from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import base64
import pandas as pd
import io
from dash.exceptions import PreventUpdate
from backend.models import LGBMModel, LinearRegressionModel
import plotly.graph_objs as go
from shap import TreeExplainer
import shap
import plotly.express as px

# Globale Variablen für Modelle
global_trained_model = None
global_test_data = None
global_prediction = None


app = Dash(__name__)
app.layout = html.Div(
    [
        html.Div(
            dcc.Upload(
                id="upload-training-data",
                children=html.Div(["Drag and Drop or Select Training Files"]),
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
        ),
        html.Div(
            dcc.Upload(
                id="upload-test-data",
                children=html.Div(["Drag and Drop or Select Test Files"]),
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
            style={"width": "49%", "display": "inline-block"},
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
                        html.Button("Train Model", id="train-model-button"),
                        dcc.Graph(
                            id="model-performance-graph",
                            config={"staticPlot": False},  # Ermöglicht Interaktion
                            style={
                                "width": "100%",
                                "height": "400px",
                            },  # Passen Sie den Stil nach Bedarf an
                        ),
                        html.Div(id="train-model-output"),  # Output für Trainingsstatus
                    ],
                    style={"width": "40%", "display": "inline-block"},
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
                        html.Div(id="shap-output", style={"marginTop": "20px"}),
                        html.Div(
                            [
                                dcc.Graph(id="shap-plot"),
                            ],
                            style={"width": "100%", "display": "inline-block"},
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
def show_upload_config_button(selected_model):
    if selected_model == "LGBM":
        return {"display": "inline-block"}
    return {"display": "none"}


@app.callback(
    Output("upload-training-data", "children"),
    [Input("upload-training-data", "contents")],
    [State("upload-training-data", "filename")],
)
def update_upload_component(contents, filename):
    if contents is not None:
        # Überprüfen Sie, ob die Datei eine CSV-Datei ist
        if filename.endswith(".csv"):
            content_type, content_string = contents.split(",")
            decoded = base64.b64decode(content_string)
            try:
                df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
                return html.Div([f"{filename} - Drag and Drop or Select for changing"])
            except Exception as e:
                return html.Div(["There was an error processing this file."])
        else:
            return html.Div(["Please upload a CSV file."])
    else:
        # Wenn keine Inhalte vorhanden sind, zeigen Sie den Standardtext an
        return html.Div(["Drag and Drop or Select Training Files"])


@app.callback(
    Output("upload-test-data", "children"),
    [Input("upload-test-data", "contents")],
    [State("upload-test-data", "filename")],
)
def update_test_upload_component(contents, filename):
    if contents is not None:
        # Überprüfen Sie, ob die Datei eine CSV-Datei ist
        if filename.endswith(".csv"):
            content_type, content_string = contents.split(",")
            decoded = base64.b64decode(content_string)
            try:
                df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
                return html.Div([f"{filename} - Drag and Drop or Select for changing"])
            except Exception as e:
                return html.Div(["There was an error processing this file."])
        else:
            return html.Div(["Please upload a CSV file."])
    else:
        # Wenn keine Inhalte vorhanden sind, zeigen Sie den Standardtext an
        return html.Div(["Drag and Drop or Select Test Files"])


@app.callback(
    [
        Output("model-performance-graph", "figure"),
        Output("train-model-output", "children"),
    ],
    [Input("train-model-button", "n_clicks")],
    [
        State("upload-training-data", "contents"),
        State("upload-test-data", "contents"),
        State("model-selection-radioitems", "value"),
    ],
)
def update_graph(n_clicks, training_contents, test_contents, selected_model):
    global global_trained_model
    global global_test_data
    global global_prediction

    if n_clicks is None:
        raise PreventUpdate

    # Überprüfen, ob sowohl Trainings- als auch Testdaten hochgeladen wurden
    if not training_contents or not test_contents:
        return html.Div("Please upload both training and test data.")

    # Datenverarbeitung
    train_df = parse_contents(training_contents)
    test_df = parse_contents(test_contents)
    test_index = test_df.date
    train_df = train_df.drop(columns=["date"])
    test_df = test_df.drop(columns=["date"])

    # Modellauswahl und Training
    if selected_model == "LGBM":
        model = LGBMModel(train_df, test_df)
        model.fit()
        predictions = model.predict()
    elif selected_model == "LR":
        model = LinearRegressionModel(train_df, test_df)
        model.fit()
        predictions = model.predict()
    else:
        return html.Div("Unknown model selected.")

    predictions_df = pd.DataFrame({"date": test_index, "yhat": predictions})
    global_trained_model = model
    global_test_data = test_df
    global_prediction = predictions_df
    predictions_plot = create_prediction_plot(predictions_df)

    return predictions_plot, html.Div("Model trained and predictions made.")


def parse_contents(contents):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    return pd.read_csv(io.StringIO(decoded.decode("utf-8")))


def create_prediction_plot(predictions_df):
    trace = go.Scatter(
        x=predictions_df["date"],
        y=predictions_df["yhat"],
        mode="lines",
        name="Predictions",
    )
    layout = go.Layout(
        title="Model Predictions",
        xaxis={"title": "Date"},
        yaxis={"title": "Predicted Value"},
    )
    return {"data": [trace], "layout": layout}


@app.callback(
    Output("shap-output", "children"),
    [Input("model-performance-graph", "clickData")],
    [State("model-selection-radioitems", "value")],
)
def display_shap_analysis(clickData, selected_model):
    if global_trained_model is None:
        return "Train a model first."
    
    if clickData is None:
        return "Click a point in the graph to analyze."


    # Erhalten Sie den Index des ausgewählten Punktes
    point_index = clickData["points"][0]["pointIndex"]

    return f"Selected point: {global_prediction.iloc[point_index].date} with prediction {global_prediction.iloc[point_index].yhat}"

@app.callback(
    Output('shap-plot', 'figure'),
    [Input('model-performance-graph', 'clickData')],
    [State('model-selection-radioitems', 'value')]
)
def display_shap_plot(clickData, selected_model):
    if clickData is None or global_trained_model is None:
        return go.Figure()

    point_index = clickData['points'][0]['pointIndex']

    if selected_model == "LGBM":
        shap_fig = create_shap_plot(global_trained_model, global_test_data, point_index)
        return shap_fig
    else:
        return go.Figure()


def create_shap_plot(model, test_data, point_index):
    # Explainer erstellen
    explainer = shap.Explainer(model.model)
    shap_values = explainer.shap_values(test_data)

    # SHAP-Werte für den ausgewählten Punkt
    selected_shap_values = shap_values[point_index]

    # SHAP-Plot erstellen
    fig = px.bar(x=test_data.columns, y=selected_shap_values, labels={'x': 'Feature', 'y': 'SHAP Value'})
    fig.update_layout(title_text='SHAP Values for Selected Prediction', title_x=0.5)
    
    return fig

if __name__ == "__main__":
    app.run_server(debug=True)
