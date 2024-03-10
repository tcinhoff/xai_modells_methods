from dash import dcc, html
from app_instance import app, PATHS
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import pandas as pd
import pickle
from ba_xai.backend.xai_methods.pdp import PDP


def get_pdp_component(selected_model, point_index):
    test_data = pd.read_csv(PATHS["processed_test_data_path"])
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

    return PDP.create_pdp_plot(model, selected_feature, test_data)


