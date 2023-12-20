from dash import dcc
import plotly.graph_objs as go
import plotly.express as px
from app_instance import PATHS
import pickle
import pandas as pd

def get_coefficents_component(selected_model, point_index):
    coef_fig = create_coefficients_plot()
    return dcc.Graph(figure=coef_fig)

def create_coefficients_plot():
    test_data = pd.read_csv(PATHS["processed_test_data_path"])
    pickled_model = open(PATHS["model_path"], "rb").read()
    model = pickle.loads(pickled_model).model
    if not hasattr(model, "coef_"):
        return go.Figure()

    coefficients = model.coef_

    fig = px.bar(
        x=test_data.columns, y=coefficients, labels={"x": "Feature", "y": "Coefficient"}
    )
    fig.update_layout(title_text="Model Coefficients", title_x=0.5)

    return fig