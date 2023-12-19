import lime
from dash import html
from app_instance import PATHS
import pickle

def get_lime_component(point_index, train_data, test_data):
    lime_html = perform_lime_analysis(
        train_data, test_data, point_index
    )
    return html.Iframe(srcDoc=lime_html, style={"width": "100%", "height": "400px"})


def perform_lime_analysis(train_data, test_data, point_index):
    print(point_index)
    pickled_model = open(PATHS.model_path, "rb").read()
    model = pickle.loads(pickled_model).model
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=train_data.to_numpy(), feature_names=train_data.columns, mode="regression"
    )
    print(test_data.iloc[point_index])
    lime_exp = explainer.explain_instance(test_data.iloc[point_index], model.predict)
    return lime_exp.as_html()
