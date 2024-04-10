from dash import html
import shap
from app_instance import PATHS
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from ba_xai.configs.models_config import MODELS


class SHAP:
    def get_shap_plot_component(selected_model, point_index, selected_method):
        image_src = SHAP.create_shap_plot(selected_model, point_index, selected_method)
        return html.Img(src=image_src, style={"width": "100%", "height": "500px"})

    def create_shap_plot(selected_model, point_index, selected_method):
        test_data = pd.read_csv(PATHS["processed_test_data_path"])
        train_data = pd.read_csv(PATHS["processed_train_data_path"])

        # Scale test_data if nessessary
        if MODELS[selected_model]["use_data_noramlization"]:
            scaler = pickle.loads(open(PATHS["path_to_scaler"], "rb").read())
            test_data = pd.DataFrame(
                scaler.transform(test_data), columns=test_data.columns
            )
            train_data = pd.DataFrame(
                scaler.transform(train_data), columns=train_data.columns
            )

        pickled_model = open(PATHS["model_path"], "rb").read()
        model = pickle.loads(pickled_model)

        # Wählen den richtigen Explainer basierend auf dem Modelltyp
        if selected_model in ["GAM", "GPR", "MLP"]:
            explainer = shap.Explainer(model.predict, train_data)
        else:
            explainer = shap.Explainer(model.model, train_data)

        if selected_method["label"] == "SHAP Global Bar for Train Data":
            shap_values = explainer(train_data, check_additivity=False)
        else:
            shap_values = explainer(test_data)

        return SHAP.create_shap_image(shap_values, point_index, selected_method)

    def create_shap_image(shap_values, point_index, selected_method):
        plt.figure(figsize=(10, 6))
        if selected_method["local"]:
            selected_method["function"](
                shap_values[point_index], max_display=14, show=False
            )
        else:
            selected_method["function"](shap_values, max_display=14, show=False)

        img_buf = BytesIO()
        plt.savefig(img_buf, format="png", bbox_inches="tight")
        plt.close()
        data = base64.b64encode(img_buf.getbuffer()).decode("ascii")

        return f"data:image/png;base64,{data}"

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
