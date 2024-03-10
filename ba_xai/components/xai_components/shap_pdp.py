from dash import dcc, html
from app_instance import app, PATHS
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import pandas as pd
import pickle
from ba_xai.backend.xai_methods.shap import SHAP


def get_shap_pdp_component():
    test_data = pd.read_csv(PATHS["processed_test_data_path"])
    return html.Div(
        [
            dcc.Dropdown(
                id="shap-pdp-feature-dropdown",
                options=[{"label": col, "value": col} for col in test_data.columns],
                value=None,
            ),
            dcc.Loading(
                id="loading-3",
                type="dot",
                children=html.Img(id="shap-pdp-image", style={"padding": "20px"}),
            ),
        ]
    )


@app.callback(
    Output("shap-pdp-image", "src"),
    [Input("shap-pdp-feature-dropdown", "value")],
)
def update_shap_pdp_plot(selected_feature):
    pickled_model = open(PATHS["model_path"], "rb").read()
    model = pickle.loads(pickled_model).model
    test_data = pd.read_csv(PATHS["processed_test_data_path"])

    if selected_feature is None or test_data is None or model is None:
        raise PreventUpdate

    return SHAP.create_shap_pdp_image(model, selected_feature, test_data)
