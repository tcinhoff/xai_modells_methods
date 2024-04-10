import lime
from app_instance import PATHS
import pickle
import pandas as pd
from ba_xai.configs.models_config import MODELS
from dash import html


class LIME:
    def get_line_html(selected_model, point_index):
        test_data = pd.read_csv(PATHS["processed_test_data_path"])
        train_data = pd.read_csv(PATHS["processed_train_data_path"])

        pickled_model = open(PATHS["model_path"], "rb").read()
        model = pickle.loads(pickled_model).model

        if MODELS[selected_model]["use_data_noramlization"]:
            with open(PATHS["path_to_scaler"], "rb") as f:
                scaler = pickle.load(f)
            scaled_train_data = scaler.transform(train_data)
            train_data = pd.DataFrame(scaled_train_data, columns=train_data.columns)
            scaled_test_data = scaler.transform(test_data)
            test_data = pd.DataFrame(scaled_test_data, columns=test_data.columns)

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=train_data.values,
            feature_names=train_data.columns,
            mode="regression",
        )

        exp = explainer.explain_instance(
            test_data.iloc[point_index].values, model.predict
        )

        return exp.as_html()

    def get_lime_Iframe(selected_model, point_index):
        lime_html = LIME.get_line_html(selected_model, point_index)
        return html.Iframe(srcDoc=lime_html, style={"width": "100%", "height": "600px"})
