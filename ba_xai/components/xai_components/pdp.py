from dash import dcc, html
import plotly.graph_objs as go
from sklearn.inspection import partial_dependence
from app_instance import app, PATHS
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
import pickle
import io


def get_pdp_component(test_data):
    return html.Div(
        [
            dcc.Dropdown(
                id="pdp-feature-dropdown",
                options=[{"label": col, "value": col} for col in test_data.columns],
                value=None,
            ),
            dcc.Graph(id="pdp-graph"),
        ]
    )


@app.callback(
    Output("pdp-graph", "figure"),
    [Input("pdp-feature-dropdown", "value")],
)
def update_pdp_plot(selected_feature):
    pickled_model = open(PATHS["model_path"], "rb").read()
    model = pickle.loads(pickled_model).model
    test_data = pd.read_csv(PATHS["processed_test_data_path"])

    if selected_feature is None or test_data is None or model is None:
        raise PreventUpdate
    
    return create_pdp_plot(model, selected_feature, test_data)


def create_pdp_plot(model, feature, data):
    pdp_results = partial_dependence(model, data, [feature])
    feature_values = pdp_results["values"][0]
    pdp_values = pdp_results["average"][0]

    trace = go.Scatter(x=feature_values, y=pdp_values, mode="lines", name="PDP")
    layout = go.Layout(
        title=f"Partial Dependence Plot for {feature}",
        xaxis={"title": feature},
        yaxis={"title": "Partial Dependence"},
    )
    return {"data": [trace], "layout": layout}
