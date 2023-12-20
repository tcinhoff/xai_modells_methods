from dash import dcc
import plotly.graph_objs as go
import plotly.express as px
import shap
from app_instance import PATHS
import pickle
from backend.models.models_config import MODELS
import pandas as pd


def get_shap_component(selected_model, point_index):
    if "SHAP" in MODELS[selected_model]["compatible_methods"]:
        shap_fig = create_shap_plot(
            selected_model, point_index
        )
        return dcc.Graph(figure=shap_fig)
    else:
        return "SHAP analysis is not available for this model."
    


def create_shap_plot(selected_model, point_index):
    test_data = pd.read_csv(PATHS["processed_test_data_path"])

    pickled_model = open(PATHS["model_path"], "rb").read()
    model = pickle.loads(pickled_model).model
    # Wählen Sie den richtigen Explainer basierend auf dem Modelltyp
    if selected_model in ["LGBM", "XGBoost"]:
        explainer = shap.TreeExplainer(model)
    elif selected_model == ["GAM", "LR"]:
        explainer = shap.Explainer(model, test_data)
    else:
        return go.Figure()  #

    shap_values = explainer.shap_values(test_data)

    selected_shap_values = shap_values[point_index]

    fig = px.bar(
        x=test_data.columns,
        y=selected_shap_values,
        labels={"x": "Feature", "y": "SHAP Value"},
    )
    fig.update_layout(title_text="SHAP Values for Selected Prediction", title_x=0.5)

    return fig
