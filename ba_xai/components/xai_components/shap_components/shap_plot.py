from dash import html
import shap
from app_instance import PATHS
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from backend.models.models_config import MODELS


def get_shap_plot_component(selected_model, point_index, selected_method):
    image_src = create_shap_plot(selected_model, point_index, selected_method)
    return html.Img(src=image_src, style={"width": "100%", "height": "500px"})


def create_shap_plot(selected_model, point_index, selected_method):
    test_data = pd.read_csv(PATHS["processed_test_data_path"])
    # Scale test_data if nessessary
    if MODELS[selected_model]["use_data_noramlization"]:
        scaler = pickle.loads(open(PATHS["path_to_scaler"], "rb").read())
        test_data = pd.DataFrame(scaler.transform(test_data), columns=test_data.columns)

    pickled_model = open(PATHS["model_path"], "rb").read()
    model = pickle.loads(pickled_model)

    # Wählen Sie den richtigen Explainer basierend auf dem Modelltyp
    if selected_model == "GAM":
        explainer = shap.Explainer(model.predict, test_data)
    else:
        explainer = shap.Explainer(model.model, test_data)
    shap_values = explainer(test_data)
    return create_shap_image(shap_values, point_index, selected_method)


def create_shap_image(shap_values, point_index, selected_method):
    plt.figure(figsize=(10, 6))
    if selected_method["local"]:
        selected_method["function"](
            shap_values[point_index], max_display=14, show=False
        )
    else:
        selected_method["function"](shap_values, max_display=14, show=False)

    # Speichern des Plots als Bild im Speicher
    img_buf = BytesIO()
    plt.savefig(img_buf, format="png", bbox_inches="tight")
    plt.close()
    data = base64.b64encode(img_buf.getbuffer()).decode("ascii")

    return f"data:image/png;base64,{data}"
