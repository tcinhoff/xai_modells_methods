from dash import dcc
import plotly.graph_objs as go
import plotly.express as px
import shap
from app_instance import model_path
import pickle


def get_shap_component(selected_model, point_index, test_data):
    if selected_model in ["LGBM", "XGBoost", "LR", "GAM"]:
        shap_fig = create_shap_plot(
            test_data, point_index, selected_model
        )
        return dcc.Graph(figure=shap_fig)
    else:
        return "SHAP analysis is not available for this model."
    


def create_shap_plot(test_data, point_index, model_type):
    pickled_model = open(model_path, "rb").read()
    model = pickle.loads(pickled_model).model
    # Wählen Sie den richtigen Explainer basierend auf dem Modelltyp
    if model_type in ["LGBM", "XGBoost"]:
        explainer = shap.TreeExplainer(model)
    elif model_type == ["GAM", "LR"]:
        explainer = shap.Explainer(model, test_data)
    else:
        return go.Figure()  # Für unbekannte Modelle keine SHAP-Plot

    shap_values = explainer.shap_values(test_data)

    # Verwenden Sie SHAP-Werte für den ausgewählten Punkt
    selected_shap_values = shap_values[point_index]

    # SHAP-Plot erstellen
    fig = px.bar(
        x=test_data.columns,
        y=selected_shap_values,
        labels={"x": "Feature", "y": "SHAP Value"},
    )
    fig.update_layout(title_text="SHAP Values for Selected Prediction", title_x=0.5)

    return fig


