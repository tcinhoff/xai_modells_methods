from dash import dcc
import plotly.graph_objs as go
import plotly.express as px
from app_instance import PATHS
import pickle

def get_coefficents_component(features):
    coef_fig = create_coefficients_plot(features)
    return dcc.Graph(figure=coef_fig)

def create_coefficients_plot( features):
    pickled_model = open(PATHS["model_path"], "rb").read()
    model = pickle.loads(pickled_model).model
    if not hasattr(model, "coef_"):
        return go.Figure()

    coefficients = model.coef_

    fig = px.bar(
        x=features, y=coefficients, labels={"x": "Feature", "y": "Coefficient"}
    )
    fig.update_layout(title_text="Model Coefficients", title_x=0.5)

    return fig