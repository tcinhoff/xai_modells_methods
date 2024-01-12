from dash import html
import shap
from app_instance import PATHS
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from backend.models.models_config import MODELS


def get_shap_model_bar_component(selected_model, point_index):
    if "SHAP" in MODELS[selected_model]["compatible_methods"]:
        image_src = create_shap_model_bar_plot(selected_model, point_index)
        return html.Img(src=image_src, style={"width": "100%", "height": "500px"})
    else:
        return "SHAP analysis is not available for this model."


def create_shap_model_bar_plot(selected_model, point_index):
    test_data = pd.read_csv(PATHS["processed_test_data_path"])
    # Scale test_data if nessessary
    if MODELS[selected_model]["use_data_noramlization"]:
        scaler = pickle.loads(open(PATHS["path_to_scaler"], "rb").read())
        test_data = pd.DataFrame(scaler.transform(test_data), columns=test_data.columns)

    pickled_model = open(PATHS["model_path"], "rb").read()
    model = pickle.loads(pickled_model).model

    # Wählen Sie den richtigen Explainer basierend auf dem Modelltyp
    explainer = shap.Explainer(model, test_data)
    shap_values = explainer(test_data)
    return create_shap_model_bar_image(shap_values)


def create_shap_model_bar_image(shap_values):
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=14, show=False)

    # Speichern des Plots als Bild im Speicher
    img_buf = BytesIO()
    plt.savefig(img_buf, format="png", bbox_inches="tight")
    plt.close()
    data = base64.b64encode(img_buf.getbuffer()).decode("ascii")

    return f"data:image/png;base64,{data}"
