from dash import dcc, html
import shap
from app_instance import app, PATHS
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import base64
from io import BytesIO


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

    return create_shap_pdp_image(model, selected_feature, test_data)


def create_shap_pdp_image(model, feature, data):
    shap.partial_dependence_plot(
        feature,
        model.predict,
        data,
        ice=False,
        model_expected_value=True,
        feature_expected_value=True,
        show=False,
    )

    plt.tight_layout()

    img_buf = BytesIO()
    plt.savefig(img_buf, format="png")
    plt.close()
    data = base64.b64encode(img_buf.getbuffer()).decode("ascii")

    return f"data:image/png;base64,{data}"
